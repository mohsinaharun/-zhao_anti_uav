# =============================================================================
# Phase 4 agents (PREREG §4-5, Amendment 1). Trained ONLY in the proxy simulator
# (fl_env.build_data('sim')): zero client privacy cost.
#   A4 blind DRL   : DDPG, state = (t/R, remaining budget fraction)
#   A5 feedback DRL: TD3 + GRU over rounds, state adds server-side PUBLIC-proxy signals
#                    (proxy loss, proxy median per-sample gradient norm, change in loss)
# Action a in [0,1] -> spend share 0.25 * 16^a of (remaining budget / rounds left).
# Reward = simulator proxy-val mAP@0.5 of the final model (sparse, undiscounted).
#   python phase4_agents.py --arm A5 --op P [--episodes 300]
# =============================================================================
import os, sys, json, time, copy, argparse
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C, budget, fl_env, phase4 as P4

DEV = C.DEV
OUT = os.path.join(C.RESULTS, 'phase4')
WARMUP, STD0, STD1 = 20, 0.3, 0.05
EVAL_EVERY, EVAL_SEEDS = 25, (900, 901, 902)


def features(obs, t, R, spent):
    """Per-round input vector. Blind agents use the first two entries only."""
    o = obs[-1]
    return [t / R, 1.0 - spent, o['proxy_loss'] / 3.0, float(np.log(max(o['proxy_gnorm'], 1e-6)) / 4.0),
            o['proxy_dloss'] * 10.0]


def mlp(i, o, h=64, out_act=None):
    layers = [nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o)]
    return nn.Sequential(*layers, *([out_act] if out_act else []))


# ── A4: DDPG on the blind state ──────────────────────────────────────────────
class DDPG:
    NIN = 2

    def __init__(self, seed):
        torch.manual_seed(seed)
        self.actor = mlp(self.NIN, 1, out_act=nn.Sigmoid()).to(DEV)
        self.critic = mlp(self.NIN + 1, 1).to(DEV)
        self.ta, self.tc = copy.deepcopy(self.actor), copy.deepcopy(self.critic)
        self.oa = torch.optim.Adam(self.actor.parameters(), 1e-3)
        self.oc = torch.optim.Adam(self.critic.parameters(), 1e-3)
        self.buf = []

    def policy(self):
        return BlindPolicy(self.actor)

    def store(self, ep):
        X, A, r = ep['X'], ep['A'], ep['reward']
        for t in range(len(A)):
            last = t == len(A) - 1
            self.buf.append((X[t][:self.NIN], A[t], r if last else 0.0, X[t + 1][:self.NIN] if not last else X[t][:self.NIN], last))

    def update(self, n, bs=64, tau=0.01):
        if len(self.buf) < bs:
            return
        for _ in range(n):
            b = [self.buf[i] for i in np.random.randint(len(self.buf), size=bs)]
            s = torch.tensor([x[0] for x in b], dtype=torch.float32, device=DEV)
            a = torch.tensor([[x[1]] for x in b], dtype=torch.float32, device=DEV)
            r = torch.tensor([[x[2]] for x in b], dtype=torch.float32, device=DEV)
            s2 = torch.tensor([x[3] for x in b], dtype=torch.float32, device=DEV)
            d = torch.tensor([[float(x[4])] for x in b], device=DEV)
            with torch.no_grad():
                y = r + (1 - d) * self.tc(torch.cat([s2, self.ta(s2)], 1))
            lc = ((self.critic(torch.cat([s, a], 1)) - y) ** 2).mean()
            self.oc.zero_grad(); lc.backward(); self.oc.step()
            la = -self.critic(torch.cat([s, self.actor(s)], 1)).mean()
            self.oa.zero_grad(); la.backward(); self.oa.step()
            for net, tgt in ((self.actor, self.ta), (self.critic, self.tc)):
                for p, q in zip(net.parameters(), tgt.parameters()):
                    q.data.mul_(1 - tau).add_(tau * p.data)


class BlindPolicy:
    def __init__(self, actor):
        self.actor = actor

    def reset(self):
        pass

    def act(self, obs, t, R, spent, explore_std=0.0):
        with torch.no_grad():
            a = float(self.actor(torch.tensor([features(obs, t, R, spent)[:2]], dtype=torch.float32, device=DEV)))
        return float(np.clip(a + np.random.normal(0, explore_std), 0, 1)) if explore_std else a


# ── A5: TD3 + GRU on the feedback state ──────────────────────────────────────
class GRUActor(nn.Module):
    def __init__(self, nin=5, h=32):
        super().__init__()
        self.gru = nn.GRU(nin, h, batch_first=True)
        self.head = mlp(h, 1, out_act=nn.Sigmoid())

    def forward(self, X):                      # X: batch x T x nin -> batch x T x 1
        return self.head(self.gru(X)[0])


class GRUCritic(nn.Module):
    def __init__(self, nin=5, h=32):
        super().__init__()
        self.gru = nn.GRU(nin, h, batch_first=True)
        self.head = mlp(h + 1, 1)

    def forward(self, X, A):                   # A: batch x T x 1
        return self.head(torch.cat([self.gru(X)[0], A], 2))


class TD3GRU:
    def __init__(self, seed):
        torch.manual_seed(seed)
        self.actor = GRUActor().to(DEV)
        self.q1, self.q2 = GRUCritic().to(DEV), GRUCritic().to(DEV)
        self.ta, self.t1, self.t2 = (copy.deepcopy(m) for m in (self.actor, self.q1, self.q2))
        self.oa = torch.optim.Adam(self.actor.parameters(), 1e-3)
        self.oq = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), 1e-3)
        self.buf, self.it = [], 0

    def policy(self):
        return GRUPolicy(self.actor)

    def store(self, ep):
        self.buf.append(ep)

    def update(self, n, bs=8, tau=0.01, noise=0.1, clip=0.2, delay=2):
        if len(self.buf) < bs:
            return
        for _ in range(n):
            eps = [self.buf[i] for i in np.random.randint(len(self.buf), size=bs)]
            X = torch.tensor([e['X'][:-1] for e in eps], dtype=torch.float32, device=DEV)    # bs x T x 5
            X2 = torch.tensor([e['X'][1:] for e in eps], dtype=torch.float32, device=DEV)
            A = torch.tensor([e['A'] for e in eps], dtype=torch.float32, device=DEV).unsqueeze(2)
            r = torch.tensor([e['reward'] for e in eps], dtype=torch.float32, device=DEV)
            T = A.shape[1]
            with torch.no_grad():
                a2 = (self.ta(X2) + (torch.randn_like(A) * noise).clamp(-clip, clip)).clamp(0, 1)
                q2 = torch.min(self.t1(X2, a2), self.t2(X2, a2)).squeeze(2)                  # bs x T
                y = torch.zeros(len(eps), T, device=DEV)
                y[:, :-1] = q2[:, :-1]                                                      # gamma = 1
                y[:, -1] = r
            lq = ((self.q1(X, A).squeeze(2) - y) ** 2).mean() + ((self.q2(X, A).squeeze(2) - y) ** 2).mean()
            self.oq.zero_grad(); lq.backward(); self.oq.step()
            self.it += 1
            if self.it % delay == 0:
                la = -self.q1(X, self.actor(X)).mean()
                self.oa.zero_grad(); la.backward(); self.oa.step()
                for net, tgt in ((self.actor, self.ta), (self.q1, self.t1), (self.q2, self.t2)):
                    for p, q in zip(net.parameters(), tgt.parameters()):
                        q.data.mul_(1 - tau).add_(tau * p.data)


class GRUPolicy:
    def __init__(self, actor):
        self.actor = actor

    def reset(self):
        self.seq = []

    def act(self, obs, t, R, spent, explore_std=0.0):
        self.seq.append(features(obs, t, R, spent))
        with torch.no_grad():
            a = float(self.actor(torch.tensor([self.seq], dtype=torch.float32, device=DEV))[0, -1, 0])
        return float(np.clip(a + np.random.normal(0, explore_std), 0, 1)) if explore_std else a


def load_policy(path, arm):
    net = (mlp(DDPG.NIN, 1, out_act=nn.Sigmoid()) if arm == 'A4' else GRUActor()).to(DEV)
    net.load_state_dict(torch.load(path, map_location=DEV)); net.eval()
    return BlindPolicy(net) if arm == 'A4' else GRUPolicy(net)


# ── one simulator episode ────────────────────────────────────────────────────
def episode(data, seed, pol, op, explore_std=0.0, random_actions=False):
    R, _ = P4.OPS[op]
    tgt = P4.target_eps(op, data['qs'])
    env = fl_env.FLEnv(data, seed); obs = [env.reset()]
    pol.reset(); X, A = [], []
    for t in range(R):
        spent = budget.eps_worst(env.sigmas, data['qs']) / tgt if env.sigmas else 0.0
        X.append(features(obs, t, R, spent))
        a = float(np.random.uniform()) if random_actions else pol.act(obs, t, R, spent, explore_std)
        if random_actions:
            pol.act(obs, t, R, spent)                     # keep the policy's sequence in step
        A.append(a)
        sig = budget.sigma_for_round(env.sigmas, 0.25 * 16 ** a, R - t, tgt, data['qs'])
        obs.append(env.step(sig))
    X.append(features(obs, R, R, budget.eps_worst(env.sigmas, data['qs']) / tgt))
    return {'X': X, 'A': A, 'reward': env.evaluate(), 'sigmas': env.sigmas, 'seed': seed}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['A4', 'A5'], required=True)
    ap.add_argument('--op', choices=list(P4.OPS), required=True)
    ap.add_argument('--episodes', type=int, default=300)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tag = 'agent_%s_%s' % (a.arm, a.op)
    logf, ckdir = os.path.join(OUT, tag + '_log.json'), os.path.join(C.CACHE, 'phase4_ckpt', tag)
    os.makedirs(ckdir, exist_ok=True)
    np.random.seed(7); torch.manual_seed(7)
    agent = DDPG(7) if a.arm == 'A4' else TD3GRU(7)
    data = fl_env.build_data('sim')
    R = P4.OPS[a.op][0]
    log = {'arm': a.arm, 'op': a.op, 'episodes': [], 'evals': [], 'meta': C.meta()}
    t0 = time.time()
    for e in range(1, a.episodes + 1):
        std = STD0 + (STD1 - STD0) * min(1.0, (e - 1) / max(a.episodes - 1, 1))
        ep = episode(data, 10000 + e, agent.policy(), a.op, explore_std=std, random_actions=e <= WARMUP)
        agent.store(ep)
        agent.update(2 * R)
        log['episodes'].append({'ep': e, 'reward': ep['reward'], 'actions': ep['A'], 'sigmas': ep['sigmas'],
                                'std': std, 'seconds': time.time() - t0})
        if e % EVAL_EVERY == 0:
            pol = agent.policy()
            evs = [episode(data, s, pol, a.op) for s in EVAL_SEEDS]
            torch.save(agent.actor.state_dict(), os.path.join(ckdir, 'ep%03d.pt' % e))
            log['evals'].append({'ep': e, 'mean_reward': float(np.mean([x['reward'] for x in evs])),
                                 'rewards': [x['reward'] for x in evs], 'sigmas': [x['sigmas'] for x in evs]})
            print('%s ep %3d  train reward (last 25) %.4f  eval reward %.4f  sigma %s  [%.0f min]'
                  % (tag, e, np.mean([x['reward'] for x in log['episodes'][-25:]]), log['evals'][-1]['mean_reward'],
                     ' '.join('%.2f' % s for s in evs[0]['sigmas']), (time.time() - t0) / 60), flush=True)
        C.save_json(log, logf)
    best = max(log['evals'], key=lambda x: x['mean_reward'])
    import shutil
    shutil.copy(os.path.join(ckdir, 'ep%03d.pt' % best['ep']), os.path.join(OUT, tag + '_best.pt'))
    log['selected_checkpoint'] = best['ep']
    first = [x['reward'] for x in log['episodes'][:25]]; last = [x['reward'] for x in log['episodes'][-25:]]
    from scipy import stats
    diff = np.mean(last) - np.mean(first)
    se = np.sqrt(np.var(first, ddof=1) / 25 + np.var(last, ddof=1) / 25)
    log['G4b'] = {'first25': float(np.mean(first)), 'last25': float(np.mean(last)), 'diff': float(diff),
                  'ci95': [float(diff - 1.96 * se), float(diff + 1.96 * se)], 'learned': bool(diff - 1.96 * se > 0)}
    C.save_json(log, logf)
    print('%s done: selected checkpoint ep %d (eval %.4f); G4b %s' % (tag, best['ep'], best['mean_reward'], log['G4b']))
