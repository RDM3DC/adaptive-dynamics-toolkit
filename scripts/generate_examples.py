"""Generate example notebooks for the Adaptive Dynamics Toolkit.

Run:
    python scripts/generate_examples.py

Creates (or overwrites) notebooks under ./examples/ :
  - pi_geodesics_toy.ipynb
  - arp_vs_adam_rosenbrock.ipynb
  - arp_two_moons.ipynb
  - sim_adaptive_gravity_rotation_curve.ipynb
  - sim_beams_trajectory.ipynb
  - compress_text_atc.ipynb
  - compress_curve_cmc.ipynb
  - tsp_toolpath_demo.ipynb

Some notebooks include placeholder implementations where the underlying
module is not yet implemented in the library. These cells are clearly
commented so they can be updated later without breaking generation.
"""
from __future__ import annotations

from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples"
EX.mkdir(parents=True, exist_ok=True)

def _nb(cells):  # noqa: ANN201 small local helper
    n = nbf.v4.new_notebook()
    n.cells = cells
    return n

def _write(path: Path, cells) -> None:  # noqa: ANN401 heterogeneous list
    with path.open("w") as f:
        nbf.write(_nb(cells), f)

# 1) Adaptive π — Toy geodesics
_write(
    EX / "pi_geodesics_toy.ipynb",
    [
        nbf.v4.new_markdown_cell(
            "# Adaptive π Geodesics (Toy)\n" "A minimal geodesic-like demo under a curvature field using Adaptive π."),
        nbf.v4.new_code_cell(
            """import numpy as np, matplotlib.pyplot as plt
from adaptive_dynamics.pi import AdaptivePi

def k_field(x,y): return 0.35*np.exp(-0.8*(x**2+y**2)) - 0.05
pi_a = AdaptivePi(curvature_fn=k_field)

A = np.array([-1.2,-0.6]); B = np.array([1.1,0.7])
line = np.stack([np.linspace(A[0],B[0],200), np.linspace(A[1],B[1],200)],1)

p = A.copy(); path=[p.copy()]
for _ in range(350):
    v = B - p; v = v/np.linalg.norm(v)
    k = k_field(p[0],p[1])
    rot = np.array([[np.cos(0.7*k),-np.sin(0.7*k)],[np.sin(0.7*k),np.cos(0.7*k)]])
    v = rot @ v
    step = 0.01 * (pi_a.pi_a(p[0],p[1]) / np.pi)
    p = p + step*v; path.append(p.copy())
path = np.array(path)

plt.plot(line[:,0],line[:,1],'--',label='straight A→B')
plt.plot(path[:,0],path[:,1],label='toy geodesic'); plt.scatter([A[0],B[0]],[A[1],B[1]])
plt.gca().set_aspect('equal','box'); plt.legend(); plt.show()"""),
    ],
)

# 2) ARP vs Adam — Rosenbrock
_write(
    EX / "arp_vs_adam_rosenbrock.ipynb",
    [
        nbf.v4.new_markdown_cell("# ARP vs Adam on Rosenbrock (PyTorch)"),
        nbf.v4.new_code_cell(
            """import torch, matplotlib.pyplot as plt
from adaptive_dynamics.arp.optimizers import ARP
device='cuda' if torch.cuda.is_available() else 'cpu'

def rosenbrock(x,a=1.0,b=100.0): return (a-x[...,0])**2 + b*(x[...,1]-x[...,0]**2)**2

def run(opt_name='adam', steps=200, lr=1e-2, alpha=0.02, mu=0.002):
    x = torch.tensor([-1.5,1.5],dtype=torch.float32,requires_grad=True,device=device)
    opt = torch.optim.Adam([x], lr=lr) if opt_name=='adam' else ARP([x], lr=lr, alpha=alpha, mu=mu)
    hist=[]
    for _ in range(steps):
        opt.zero_grad(); loss = rosenbrock(x); loss.backward(); opt.step(); hist.append(loss.item())
    return hist

adam = run('adam', lr=0.01)
arp  = run('arp',  lr=0.05)
plt.yscale('log'); plt.plot(adam,label='Adam'); plt.plot(arp,label='ARP')
plt.title('Rosenbrock loss (log)'); plt.xlabel('step'); plt.ylabel('loss'); plt.legend(); plt.show()"""),
    ],
)

# 3) ARP — Two Moons classifier
_write(
    EX / "arp_two_moons.ipynb",
    [
        nbf.v4.new_markdown_cell("# ARP on Synthetic Two-Moons (PyTorch)"),
        nbf.v4.new_code_cell(
            """import torch, math, matplotlib.pyplot as plt
from adaptive_dynamics.arp.optimizers import ARP
device='cuda' if torch.cuda.is_available() else 'cpu'

def two_moons(n=200, noise=0.05):
    t = torch.rand(n)*math.pi
    x1 = torch.stack([torch.cos(t), torch.sin(t)],1)
    x2 = torch.stack([1-torch.cos(t), 1-torch.sin(t)-0.5],1)
    X = torch.cat([x1,x2],0) + noise*torch.randn(2*n,2)
    y = torch.cat([torch.zeros(n,dtype=torch.long), torch.ones(n,dtype=torch.long)],0)
    return X,y

X,y = two_moons(400,0.08); plt.scatter(X[:,0],X[:,1],c=y,s=8); plt.title('Two Moons'); plt.show()
model = torch.nn.Sequential(torch.nn.Linear(2,32),torch.nn.ReLU(),torch.nn.Linear(32,2)).to(device)
opt = ARP(model.parameters(), lr=5e-3, alpha=0.02, mu=0.002); loss_fn = torch.nn.CrossEntropyLoss()
X,y = X.to(device), y.to(device)

for epoch in range(1,51):
    logits = model(X); loss = loss_fn(logits,y)
    loss.backward(); opt.step(); model.zero_grad(set_to_none=True)
    if epoch%10==0:
        acc = (logits.argmax(1)==y).float().mean().item()
        print(f'epoch {epoch} loss={loss.item():.4f} acc={acc:.3f}')"""),
    ],
)

# 4) Adaptive gravity — rotation curve (placeholder)
_write(
    EX / "sim_adaptive_gravity_rotation_curve.ipynb",
    [
        nbf.v4.new_markdown_cell("# Adaptive Gravity — Rotation Curve Demo"),
        nbf.v4.new_code_cell(
            """# Placeholder example — gravity module not implemented yet.
import numpy as np, matplotlib.pyplot as plt
r = np.linspace(0.2, 8.0, 200)
v = 0.8 * (1 - np.exp(-0.4*r)) / np.sqrt(r)
plt.plot(r, v, label='Adaptive v(r) (toy)')
plt.title('Adaptive rotation curve (toy)'); plt.xlabel('r'); plt.ylabel('v'); plt.legend(); plt.show()"""),
    ],
)

# 5) Adaptive beams — trajectory (placeholder)
_write(
    EX / "sim_beams_trajectory.ipynb",
    [
        nbf.v4.new_markdown_cell("# Adaptive Charged-Particle Beam (Toy)"),
        nbf.v4.new_code_cell(
            """# Placeholder example — beams module not implemented.
import numpy as np, matplotlib.pyplot as plt
theta = np.linspace(0, 10*np.pi, 1500)
x = 0.01*theta * np.cos(theta)
y = 0.01*theta * np.sin(theta)
plt.plot(x, y); plt.title('Adaptive beam trajectory (toy)')
plt.xlabel('x'); plt.ylabel('y'); plt.show()"""),
    ],
)

# 6) ATC — text compression (placeholder)
_write(
    EX / "compress_text_atc.ipynb",
    [
        nbf.v4.new_markdown_cell("# Adaptive Text Compression (ATC)"),
        nbf.v4.new_code_cell(
            """# Placeholder until compression.atc is implemented.
s = ('In curved spaces, constants may adapt locally. ' * 4)
enc = s.encode('utf-8')  # dummy stand-in
dec = enc.decode('utf-8')
ratio = len(enc)/len(s)
print('Encoded bytes:', len(enc), 'Raw chars:', len(s), 'Ratio:', round(ratio,3))
print('Roundtrip OK:', dec == s)"""),
    ],
)

# 7) CMC — curve compression spiral (placeholder)
_write(
    EX / "compress_curve_cmc.ipynb",
    [
        nbf.v4.new_markdown_cell("# Adaptive Curve Compression (CMC) — Spiral Demo"),
        nbf.v4.new_code_cell(
            """# Placeholder until compression.cmc module exists.
import numpy as np, matplotlib.pyplot as plt
t = np.linspace(0, 6*np.pi, 600)
curve = np.stack([0.1*t*np.cos(t), 0.1*t*np.sin(t)],1)
rec = curve + 0.0005*np.random.default_rng(0).normal(size=curve.shape)
plt.plot(curve[:,0], curve[:,1], label='orig'); plt.plot(rec[:,0], rec[:,1], label='recon', alpha=0.7)
plt.title('CMC spiral encode/decode (placeholder)'); plt.legend(); plt.show()"""),
    ],
)

# 8) TSP toolpath demo (placeholder)
_write(
    EX / "tsp_toolpath_demo.ipynb",
    [
        nbf.v4.new_markdown_cell("# TSP Toolpath Demo"),
        nbf.v4.new_code_cell(
            """# Placeholder until tsp optimization utilities implemented.
import numpy as np, matplotlib.pyplot as plt
rng = np.random.default_rng(0); pts = rng.uniform(-1,1,(100,2))
order = np.arange(len(pts))  # identity path placeholder
path = pts[order]
plt.plot(path[:,0], path[:,1]); plt.title('NN TSP path (placeholder)'); plt.show()"""),
    ],
)

print("Wrote notebooks to", EX)
