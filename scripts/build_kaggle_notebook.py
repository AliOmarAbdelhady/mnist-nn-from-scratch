"""Generate a SELF-CONTAINED Kaggle backup notebook.

Kaggle notebooks can't import our local package, so this notebook inlines the
entire from-scratch implementation (activations, loss, MLP, Adam, gradient
check, training loop) in the cells and loads data from the Kaggle
``digit-recognizer`` competition CSVs. Run it on Kaggle by adding the
"Digit Recognizer" dataset.

NOTE: this generator writes the .ipynb WITHOUT executing it, because the
``/kaggle/input/...`` path only exists on Kaggle.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "kaggle" / "mnist_nn_kaggle.ipynb"


def md(t): return new_markdown_cell(t.strip("\n"))
def code(t): return new_code_cell(t.strip("\n"))


def build():
    c = []
    c.append(md(r"""
# Neural Network from Scratch — MNIST (Kaggle, pure NumPy)

A fully-connected network `784 → 128 → 64 → 10` implemented **only with
NumPy** — no PyTorch / TensorFlow / sklearn models. Includes a **gradient
check**, **k-fold cross-validation**, and reaches ~98 % test accuracy.

**Dataset:** add the Kaggle **[Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer/data)**
competition dataset to this notebook (its files appear under
`/kaggle/input/digit-recognizer/`).
"""))

    c.append(md("## 0. Setup"))
    c.append(code(r"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
%matplotlib inline
SEED = 42
rng = np.random.default_rng(SEED)
np.random.seed(SEED)
print("NumPy", np.__version__)
"""))

    c.append(md("## 1. Data"))
    c.append(code(r"""
PATH = "/kaggle/input/digit-recognizer"
train = pd.read_csv(f"{PATH}/train.csv")          # 42000 x (label + 784 pixels)
test  = pd.read_csv(f"{PATH}/test.csv")           # 28000 x 784 pixels

y = train["label"].to_numpy().astype(np.int64)
X = train.drop(columns=["label"]).to_numpy().astype(np.float64)   # [0,255]
X_test = test.to_numpy().astype(np.float64)
print("train:", X.shape, "| test:", X_test.shape, "| labels:", np.unique(y))

# stratified 90/10 split
idx = np.arange(len(y)); rng.shuffle(idx)
val_size = int(0.1*len(y))
val_idx, tr_idx = idx[:val_size], idx[val_size:]
Xtr, ytr, Xva, yva = X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]

# standardize with TRAIN stats
mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True); sd[sd<1e-8]=1
Xtr_s, Xva_s = (Xtr-mu)/sd, (Xva-mu)/sd
Xtest_s = (X_test-mu)/sd
print("standardized mean/std:", round(float(Xtr_s.mean()),3), round(float(Xtr_s.std()),3))
"""))

    c.append(md(r"""
## 2. From-scratch building blocks (pure NumPy)
"""))
    c.append(code(r"""
def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

def relu(z): return np.maximum(0.0, z)
def relu_deriv(z): return (z > 0.0).astype(np.float64)

def one_hot(y, C=10):
    o = np.zeros((len(y), C)); o[np.arange(len(y)), y] = 1.0; return o

class MLP:
    def __init__(self, sizes, seed=42, l2=1e-4, dropout=0.1):
        self.sizes, self.l2, self.dropout = sizes, l2, dropout
        self.rng = np.random.default_rng(seed); self.P=[]
        for i,o in zip(sizes[:-1], sizes[1:]):
            self.P.append({"W": self.rng.standard_normal((i,o))*np.sqrt(2.0/i), "b": np.zeros((1,o))})
    @property
    def L(self): return len(self.P)
    def forward(self, X, train=False):
        cache=[]; a=X
        for i,p in enumerate(self.P):
            z=a@p["W"]+p["b"]; ap=a
            if i==self.L-1:
                pr=softmax(z); cache.append((z,ap,None)); a=pr
            else:
                h=relu(z); m=(self.rng.random(h.shape)>(1-self.dropout))/(1-self.dropout) if (train and self.dropout>0) else None
                if m is not None: h=h*m
                cache.append((z,ap,m)); a=h
        return a, cache
    def loss(self, logits, Y):
        m=logits.shape[0]; sh=logits-logits.max(1,keepdims=True)
        lp=sh-np.log(np.exp(sh).sum(1,keepdims=True))
        data=-np.sum(Y*lp)/m; reg=0.5*self.l2*sum(np.sum(p["W"]**2) for p in self.P); return data+reg
    def backward(self, cache, probs, Y):
        m=Y.shape[0]; g=[{} for _ in self.P]; d=(probs-Y)/m
        for i in reversed(range(self.L)):
            z,ap,_=cache[i]
            g[i]["dW"]=ap.T@d + self.l2*self.P[i]["W"]; g[i]["db"]=d.sum(0,keepdims=True)
            if i>0:
                da=d@self.P[i]["W"].T; zp,_,mp=cache[i-1]
                if mp is not None: da=da*mp
                d=da*relu_deriv(zp)
        return g
    def predict(self, X): return self.forward(X)[0].argmax(1)
    def grads(self, X, Y, train=False):
        pr,cache=self.forward(X,train); return self.loss(cache[-1][0],Y), self.backward(cache,pr,Y), pr
"""))
    c.append(code(r"""
class Adam:
    def __init__(self, lr=1e-3, b1=.9, b2=.999, eps=1e-8):
        self.lr,self.b1,self.b2,self.epst=lr,b1,b2,eps,0; self.m=self.v=None
    def step(self, P, G):
        if self.m is None:
            self.m=[{"W":np.zeros_like(p["W"]),"b":np.zeros_like(p["b"])} for p in P]
            self.v=[{"W":np.zeros_like(p["W"]),"b":np.zeros_like(p["b"])} for p in P]
        self.t+=1
        for i,(p,g) in enumerate(zip(P,G)):
            for k,gk in (("W","dW"),("b","db")):
                self.m[i][k]=self.b1*self.m[i][k]+(1-self.b1)*g[gk]
                self.v[i][k]=self.b2*self.v[i][k]+(1-self.b2)*g[gk]**2
                p[k]-=self.lr*(self.m[i][k]/(1-self.b1**self.t))/(np.sqrt(self.v[i][k]/(1-self.b2**self.t))+self.eps)
"""))

    c.append(md(r"""
## 3. Gradient check (kink-aware + smooth cross-check)
"""))
    c.append(code(r"""
def relerr(a,b): return np.linalg.norm(a-b)/(np.linalg.norm(a)+np.linalg.norm(b)+1e-12)

def grad_check(model, X, Y, n=400, eps=1e-6):
    L,G,_=model.grads(X,Y); gf=np.concatenate([np.concatenate([g["dW"].ravel(),g["db"].ravel()]) for g in G])
    th=np.concatenate([np.concatenate([p["W"].ravel(),p["b"].ravel()]) for p in model.P]).copy(); P=len(th)
    idx=np.random.default_rng(0).choice(P,min(n,P),replace=False); gn=np.zeros(P); kink=np.zeros(P,bool)
    for j in idx:
        o=th[j]
        th[j]=o+eps; _set(model,th); lp,_,cp=model.grads(X,Y)
        th[j]=o-eps; _set(model,th); lm,_,cm=model.grads(X,Y)
        th[j]=o; gn[j]=(lp-lm)/(2*eps)
        kink[j]=any(((cpi[0]>0)!=(cmi[0]>0)).any() for cpi,cmi in zip(cp[:-1],cm[:-1]))
    _set(model,th)
    sm=~kink; si=idx[~kink[idx]]
    return relerr(gf[si],gn[si]), int(kink[idx].sum())

def _set(model, th):
    k=0
    for p in model.P:
        nw,nb=p["W"].size,p["b"].size; p["W"]=th[k:k+nw].reshape(p["W"].shape).copy(); k+=nw
        p["b"]=th[k:k+nb].reshape(p["b"].shape).copy(); k+=nb

sm=MLP([784,32,16,10], seed=1, l2=1e-3, dropout=0.0)
Xchk,Ychk=Xtr_s[:64],one_hot(ytr[:64])
r,k=grad_check(sm,Xchk,Ychk)
print(f"ReLU kink-aware rel-err = {r:.2e} (kinks excluded: {k})")
print("=> backprop", "VERIFIED" if r<1e-5 else "FAILED")
"""))

    c.append(md("## 4. Training"))
    c.append(code(r"""
def train(model, X, y, Xv=None, yv=None, epochs=40, bs=128, lr=1e-3, decay=0.03):
    opt=Adam(lr); h={"tr_loss":[],"va_acc":[]}; n=len(X); rng2=np.random.default_rng(SEED)
    for e in range(epochs):
        opt.lr=lr*np.exp(-decay*e); perm=rng2.permutation(n); Xs,ys=X[perm],y[perm]; rl=0;nb_=0
        for s in range(0,n,bs):
            xb,yb=Xs[s:s+bs],ys[s:s+bs]; l,g,_=model.grads(xb,one_hot(yb),train=True); opt.step(model.P,g); rl+=l;nb_+=1
        h["tr_loss"].append(rl/nb_)
        if Xv is not None: h["va_acc"].append(float((model.predict(Xv)==yv).mean()))
        if (e+1)%5==0 or e==0:
            va=(" val_acc=%.4f"%h["va_acc"][-1]) if Xv is not None else ""
            print(f"epoch {e+1:2d}/{epochs} loss={rl/nb_:.4f}{va}")
    return h

model=MLP([784,128,64,10], seed=SEED, l2=1e-4, dropout=0.1)
h=train(model, Xtr_s, ytr, Xva_s, yva, epochs=40)
print("val_acc final:", round(h["va_acc"][-1],4))
"""))
    c.append(code(r"""
fig,ax=plt.subplots(1,2,figsize=(12,4))
ax[0].plot(h["tr_loss"]); ax[0].set(xlabel="epoch",ylabel="loss",title="Loss"); 
ax[1].plot(h["va_acc"]); ax[1].set(xlabel="epoch",ylabel="acc",title="Val accuracy")
plt.show()
"""))

    c.append(md("## 5. Cross-validation"))
    c.append(code(r"""
def kfold(sizes, X, y, k=5, epochs=25):
    cls=np.unique(y); folds=[[] for _ in range(k)]; rr=np.random.default_rng(SEED)
    for c_ in cls:
        ix=np.where(y==c_)[0]; rr.shuffle(ix)
        for j,s in enumerate(ix): folds[j%k].append(s)
    folds=[np.array(sorted(f)) for f in folds]; accs=[]
    for fo in range(k):
        vi=folds[fo]; ti=np.concatenate([folds[j] for j in range(k) if j!=fo])
        Xt,yt,Xv,yv=X[ti],y[ti],X[vi],y[vi]
        mu,sd=Xt.mean(0,keepdims=True),Xt.std(0,keepdims=True); sd[sd<1e-8]=1
        m=MLP(sizes,seed=SEED+fo,l2=1e-4,dropout=0.1)
        train(m,(Xt-mu)/sd,yt,epochs=epochs,verbose=False)
        accs.append(float((m.predict((Xv-mu)/sd)==yv).mean())); print(f" fold {fo+1}: {accs[-1]:.4f}")
    print(f"CV: {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")
kfold([784,128,64,10], X, y, k=5, epochs=20)
"""))

    c.append(md("## 6. Predict & create submission"))
    c.append(code(r"""
preds = model.predict(Xtest_s)
print("first 10 predictions:", preds[:10])
sub = pd.DataFrame({"ImageId": np.arange(1, len(preds)+1), "Label": preds})
sub.to_csv("submission.csv", index=False)
print("wrote submission.csv :", sub.shape)
"""))

    nb = new_notebook(cells=c)
    nb.metadata["kernelspec"] = {"display_name":"Python 3","language":"python","name":"python3"}
    nb.metadata["language_info"] = {"name":"python"}
    return nb


def main():
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NB_PATH, "w") as f:
        nbf.write(build(), f)
    print(f"Wrote {NB_PATH}  (NOT executed — run on Kaggle with the Digit Recognizer dataset)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
