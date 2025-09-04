# scripts/generate_many_examples.py
# Generates 50 parameterized example notebooks across 10 themes (5 variants each).
# Output: ./examples/*.ipynb and ./examples/README.md

from pathlib import Path
import nbformat as nbf
from textwrap import dedent
import random

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples"
EX.mkdir(parents=True, exist_ok=True)

def nb(cells):
    n = nbf.v4.new_notebook(); n.cells = cells; return n

def w(path, cells):
    with open(path, "w") as f: nbf.write(nb(cells), f)

index_rows = []

def add_index_row(fn, title, blurb):
    index_rows.append(f"- **[{title}]({fn})** — {blurb}")

# ---------- Helpers ----------

def md(t): return nbf.v4.new_markdown_cell(dedent(t).strip())

def code(t): return nbf.v4.new_code_cell(dedent(t).strip())

# ---------- THEMES (10) ----------
# Each theme returns (title, blurb, cells)

def theme_pi_field_variation(idx, curvature_amp, baseline):
    title = f"Adaptive π — Curvature Field #{idx}"
    cells = [
        md(f"""# {title}\nToy curvature field with amplitude={curvature_amp}, baseline={baseline}.\nWe visualize how `πₐ = π·(1 + 0.5·k)` changes circle circumference."""),
    code(f"""
import numpy as np, matplotlib.pyplot as plt
from adaptive_dynamics.pi import AdaptivePi
amp = {curvature_amp}; base = {baseline}

def k_field(x,y): return amp*np.exp(-0.8*(x**2+y**2)) + base
pi_a = AdaptivePi(curvature_fn=k_field)
print('flat-limit:', np.isclose(AdaptivePi().pi_a(0.0,0.0), np.pi))

r = 0.8; theta = np.linspace(0,2*np.pi,512)
scale_val = pi_a.pi_a(0.0,0.0)/np.pi
plt.figure(); plt.gca().set_aspect('equal','box')
plt.plot(r*np.cos(theta), r*np.sin(theta), '--', label='Euclidean')
plt.plot(scale_val*r*np.cos(theta), scale_val*r*np.sin(theta), label=f'Adaptive scale={{scale_val:.3f}}')
plt.legend(); plt.title('Curved vs Euclidean circles'); plt.show()
""")
    ]
    return title, "πₐ circles with varying curvature fields", cells

def theme_arp_rosenbrock(idx, steps, adam_lr, arp_lr, alpha, mu):
    title = f"ARP vs Adam on Rosenbrock #{idx}"
    cells = [
        md(f"""# {title}\nCompare ARP to Adam on the Rosenbrock function for {steps} steps."""),
        code(f"""
import torch, matplotlib.pyplot as plt
from adaptive_dynamics.arp.optimizers import ARP
device='cuda' if torch.cuda.is_available() else 'cpu'

def rosenbrock(x,a=1.0,b=100.0): return (a-x[0])**2 + b*(x[1]-x[0]**2)**2

def run(opt_name='adam', steps={steps}, lr=1e-2, alpha={alpha}, mu={mu}):
    x = torch.tensor([-1.5,1.5], dtype=torch.float32, requires_grad=True, device=device)
    opt = torch.optim.Adam([x], lr=lr) if opt_name=='adam' else ARP([x], lr=lr, alpha=alpha, mu=mu)
    hist=[]
    for _ in range(steps):
        opt.zero_grad(); loss = rosenbrock(x); loss.backward(); opt.step(); hist.append(loss.item())
    return hist

adam = run('adam', lr={adam_lr})
arp  = run('arp',  lr={arp_lr})
plt.figure(); plt.yscale('log'); plt.plot(adam,label='Adam'); plt.plot(arp,label='ARP')
plt.title('Rosenbrock loss (log)'); plt.xlabel('step'); plt.ylabel('loss'); plt.legend(); plt.show()
""")
    ]
    return title, "Non-convex benchmark comparison", cells

def theme_arp_two_moons(idx, n, noise, hidden, lr, alpha, mu):
    title = f"ARP Two-Moons Classifier #{idx}"
    cells = [
        md(f"""# {title}\nTrain a small classifier on two-moons with noise={noise}, hidden={hidden}."""),
        code(f"""
import torch, math, matplotlib.pyplot as plt
from adaptive_dynamics.arp.optimizers import ARP
device='cuda' if torch.cuda.is_available() else 'cpu'

def two_moons(n={n}, noise={noise}):
    t = torch.rand(n)*math.pi
    x1 = torch.stack([torch.cos(t), torch.sin(t)],1)
    x2 = torch.stack([1-torch.cos(t), 1-torch.sin(t)-0.5],1)
    X = torch.cat([x1,x2],0) + noise*torch.randn(2*n,2)
    y = torch.cat([torch.zeros(n,dtype=torch.long), torch.ones(n,dtype=torch.long)],0)
    return X,y

X,y = two_moons()
plt.figure(); plt.scatter(X[:,0], X[:,1], c=y, s=8); plt.title('Two Moons'); plt.show()

model = torch.nn.Sequential(
    torch.nn.Linear(2, {hidden}), torch.nn.ReLU(),
    torch.nn.Linear({hidden}, 2)
).to(device)

opt = ARP(model.parameters(), lr={lr}, alpha={alpha}, mu={mu})
loss_fn = torch.nn.CrossEntropyLoss()
X,y = X.to(device), y.to(device)

for epoch in range(1, 61):
    logits = model(X); loss = loss_fn(logits, y)
    loss.backward(); opt.step(); model.zero_grad(set_to_none=True)
    if epoch % 15 == 0:
        acc = (logits.argmax(1)==y).float().mean().item()
        print(f'epoch {{epoch}} loss={{loss.item():.4f}} acc={{acc:.3f}}')
""")
    ]
    return title, "Simple ARP training on synthetic data", cells

def theme_mnist_tiny(idx, epochs, lr, alpha, mu):
    title = f"ARP on MNIST (Tiny) #{idx}"
    cells = [
        md(f"""# {title}\nMinimal MNIST classifier using ARP for {epochs} epochs (requires torchvision)."""),
        code(f"""
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from adaptive_dynamics.arp.optimizers import ARP
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tfm = transforms.Compose([transforms.ToTensor()])
train = datasets.MNIST(root='./data', train=True,  transform=tfm, download=True)
test  = datasets.MNIST(root='./data', train=False, transform=tfm, download=True)
train_loader = DataLoader(train, batch_size=128, shuffle=True)
test_loader  = DataLoader(test,  batch_size=256)

model = nn.Sequential(nn.Flatten(), nn.Linear(28*28, 128), nn.ReLU(), nn.Linear(128,10)).to(device)
opt = ARP(model.parameters(), lr={lr}, alpha={alpha}, mu={mu})
loss_fn = nn.CrossEntropyLoss()

def eval_acc():
    model.eval(); c=0; t=0
    with torch.no_grad():
        for X,y in test_loader:
            X,y = X.to(device), y.to(device)
            p = model(X).argmax(1)
            c += (p==y).sum().item(); t += y.numel()
    return c/t

for ep in range(1, {epochs}+1):
    model.train()
    for X,y in train_loader:
        X,y = X.to(device), y.to(device)
        loss = loss_fn(model(X), y)
        loss.backward(); opt.step(); opt.zero_grad()
    print('epoch', ep, 'acc=', round(eval_acc(),3))
""")
    ]
    return title, "MNIST quickstart with ARP optimizer", cells

def theme_gravity_profiles(idx, k, r0):
    title = f"Adaptive Gravity Rotation Curve #{idx}"
    cells = [
        md(f"""# {title}\nToy rotation curve with k={k}, r0={r0}."""),
        code(f"""
import numpy as np, matplotlib.pyplot as plt
from adaptive_dynamics.sim.gravity import AdaptiveGravity, adaptive_G
g = AdaptiveGravity(lambda r: adaptive_G(r, G0=1.0, k={k}, r0={r0}))
r = np.linspace(0.2, 12.0, 300)
plt.plot(r, g.circular_velocity(r), label='Adaptive v(r)')
plt.xlabel('r'); plt.ylabel('v'); plt.title('Adaptive rotation curve (toy)'); plt.legend(); plt.show()
""")
    ]
    return title, "Astro-flavored toy model", cells

def theme_beam_trajectories(idx, alpha, beta, steps):
    title = f"Adaptive Beam Trajectory #{idx}"
    cells = [
        md(f"""# {title}\n2D charged-particle toy trajectory with alpha={alpha}, beta={beta}, steps={steps}."""),
        code(f"""
import numpy as np, matplotlib.pyplot as plt
from adaptive_dynamics.sim.beams import Beam2D
b = Beam2D(alpha={alpha}, beta={beta})
traj = b.integrate(x0=0.0, y0=0.0, vx0=0.5, vy0=0.1, steps={steps}, dt=0.01)
plt.plot(traj[:,0], traj[:,1]); plt.title('Adaptive beam trajectory (toy)')
plt.xlabel('x'); plt.ylabel('y'); plt.gca().set_aspect('equal','box'); plt.show()
""")
    ]
    return title, "Beam steering demo", cells

def theme_text_compress(idx, repeats):
    title = f"ATC Text Compression #{idx}"
    cells = [
        md(f"""# {title}\nRoundtrip encode/decode with a repeated phrase x{repeats}."""),
        code(f"""
from adaptive_dynamics.compress import atc
s = ('In curved spaces, constants may adapt locally. ' * {repeats})
enc = atc.encode_text(s); dec = atc.decode_text(enc)
ratio = len(enc)/len(s)
print('Encoded bytes:', len(enc), 'Raw chars:', len(s), 'Ratio:', round(ratio,3))
print('Roundtrip OK:', dec == s)
""")
    ]
    return title, "Toy text codec demo", cells

def theme_curve_compress(idx, quant):
    title = f"CMC Curve Compression #{idx}"
    cells = [
        md(f"""# {title}\nSpiral encode/decode with quant={quant}."""),
        code(f"""
import numpy as np, matplotlib.pyplot as plt
from adaptive_dynamics.compress.cmc import encode_curve, decode_curve
t = np.linspace(0, 6*np.pi, 600)
curve = np.stack([0.1*t*np.cos(t), 0.1*t*np.sin(t)],1)
enc = encode_curve(curve, quant={quant}); rec = decode_curve(enc)
plt.plot(curve[:,0], curve[:,1], label='orig')
plt.plot(rec[:,0], rec[:,1], label='recon', alpha=0.7)
plt.title('CMC spiral encode/decode'); plt.legend(); plt.show()
""")
    ]
    return title, "Toy curve codec demo", cells

def theme_tsp_variants(idx, npts, seed):
    title = f"TSP Toolpath Variant #{idx}"
    cells = [
        md(f"""# {title}\nNearest-neighbor tour on {npts} points (seed={seed})."""),
        code(f"""
import numpy as np, matplotlib.pyplot as plt
from adaptive_dynamics.tsp.slicer import optimize_path
from adaptive_dynamics.tsp.postproc import reorder_path
rng = np.random.default_rng({seed})
pts = rng.uniform(-1,1,({npts},2))
order = optimize_path(pts); path = pts[order]
post = reorder_path(path)
plt.plot(path[:,0], path[:,1]); plt.title('NN TSP path'); plt.show()
plt.plot(post[:,0], post[:,1]); plt.title('Post-processed path'); plt.show()
""")
    ]
    return title, "TSP demo with random point sets", cells

def theme_bench_synth_loss(idx, T, adam_lr, arp_lr, alpha, mu, noise):
    title = f"ARP vs Adam — Synthetic Quadratic #{idx}"
    cells = [
        md(f"""# {title}\nNoisy quadratic loss with T={T}, noise={noise}."""),
        code(f"""
import numpy as np, matplotlib.pyplot as plt
from numpy.random import default_rng
rng = default_rng({idx})
d = 20
eigs = np.linspace(0.5,4.0,d)
A = np.diag(eigs)
w_star = rng.normal(0,1.0,size=d)
w0 = rng.normal(0,2.0,size=d)
noise_sigma = {noise}

def loss(w):
    z = w - w_star
    return 0.5 * float(z.T @ A @ z)

def grad(w):
    z = w - w_star
    g = A @ z
    if noise_sigma>0: g = g + rng.normal(0, noise_sigma, size=g.shape)
    return g

def run_adam(T={T}, lr={adam_lr}, b1=0.9, b2=0.999, eps=1e-8):
    w = w0.copy(); m = np.zeros_like(w); v = np.zeros_like(w); hist=[]
    for t in range(1, T+1):
        g=grad(w); m=b1*m+(1-b1)*g; v=b2*v+(1-b2)*(g*g)
        m_hat=m/(1-b1**t); v_hat=v/(1-b2**t)
        w = w - lr*m_hat/(np.sqrt(v_hat)+eps)
        hist.append(loss(w))
    return np.array(hist)

def run_arp(T={T}, lr={arp_lr}, alpha={alpha}, mu={mu}):
    w = w0.copy(); G = np.zeros_like(w); hist=[]
    for _ in range(T):
        g=grad(w); G = G + alpha*np.abs(g) - mu*G; G = np.maximum(G,0.0)
        w = w - lr * g / (1.0 + G)
        hist.append(loss(w))
    return np.array(hist)

adam = run_adam(); arp = run_arp()
plt.figure(); plt.yscale('log')
plt.plot(adam, label='Adam'); plt.plot(arp, label='ARP-style')
plt.title('Synthetic quadratic (log loss)'); plt.xlabel('step'); plt.ylabel('loss'); plt.legend(); plt.show()
""")
    ]
    return title, "Synthetic optimizer benchmark", cells

# ---------- BUILD 50 NOTEBOOKS ----------
random.seed(7)
specs = []

for i in range(1, 6):
    specs.append(("pi", theme_pi_field_variation, (i, round(0.25+0.1*i,3), round(-0.07+0.01*i,3))))
    specs.append(("rosen", theme_arp_rosenbrock, (i, 200+20*i, 0.01, 0.04+0.01*i, 0.02, 0.002)))
    specs.append(("moons", theme_arp_two_moons, (i, 400+40*i, round(0.04+0.01*i,3), 24+2*i, 0.006, 0.02, 0.002)))
    specs.append(("mnist", theme_mnist_tiny, (i, 2+i, 0.003+0.001*i, 0.01, 0.001)))
    specs.append(("grav", theme_gravity_profiles, (i, round(0.10+0.02*i,3), round(0.8+0.2*i,2))))
    specs.append(("beam", theme_beam_trajectories, (i, round(0.02+0.004*i,3), round(0.008+0.002*i,3), 1200+100*i)))
    specs.append(("atc", theme_text_compress, (i, 10+5*i)))
    specs.append(("cmc", theme_curve_compress, (i, round(0.0008+0.0004*i,6))))
    specs.append(("tsp", theme_tsp_variants, (i, 80+10*i, 100+i)))
    specs.append(("bench", theme_bench_synth_loss, (i, 200+20*i, 0.05, 0.10, 0.02, 0.002, round(0.015+0.003*i,3))))

count = 0
for prefix, builder, args in specs:
    count += 1
    title, blurb, cells = builder(*args)
    slug = title.lower().replace(" ", "_").replace("—","-").replace("–","-").replace("#","")
    fn = f"{prefix}_{count:02d}_{slug}.ipynb"
    w(EX / fn, cells)
    add_index_row(fn, title, blurb)

index_md = "# Examples Index (Generated)\n\n" + "\n".join(index_rows) + "\n"
(EX / "README.md").write_text(index_md, encoding="utf-8")

print(f"Generated {count} notebooks into {EX}")
