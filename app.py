# app.py
import streamlit as st
import random
import math
import numpy as np
import pandas as pd

st.set_page_config(page_title="Optimisation — Multiplicateurs & Probabilités", layout="wide")
st.title("Recherche de solutions optimales")

# ------------------ Paramètres FIXES (non exposés) ------------------
target_sum = 10_000
target_val = 100_000

# ------------------ Choix du nombre de multiplicateurs ------------------
st.subheader("Configuration générale")
nb_multiplicateurs = st.number_input(
    "Nombre de multiplicateurs (entre 5 et 6)",
    min_value=5,
    max_value=6,
    value=6,
    step=1,
)
n = int(nb_multiplicateurs)

# ------------------ Entrées visibles ------------------
st.subheader(f"Multiplicateurs (M1..M{n})")

# Liste de base (on tronque à n). Les 6 premiers sont tes valeurs d'origine.
base_default_weights_display = [0.4, 1.0, 2.0, 3.0, 10.0, 50.0, 100.0, 200.0]
default_weights_display = base_default_weights_display[:n]

col_w = st.columns(n)
weights_display = []
for i, c in enumerate(col_w):
    with c:
        weights_display.append(
            float(
                st.number_input(
                    f"M{i+1}",
                    value=float(default_weights_display[i]),
                    step=0.1,
                    format="%.2f"
                )
            )
        )

# Conversion pour le calcul (on ×10 comme demandé)
weights = [w * 10.0 for w in weights_display]

st.subheader(f"Probabilités (P1..P{n}) — entiers / 10000")
st.caption("Vous pouvez fixer certaines probabilités, laissez libre sinon")

col_x = st.columns(n)
X_FIXED = []
for i, c in enumerate(col_x):
    with c:
        s = st.text_input(
            f"P{i+1}",
            value="",
            help="Ex: 6000 signifie 0.6000 ; laisser vide pour libre"
        )
        s = s.strip()
        if s == "":
            X_FIXED.append(None)
        else:
            try:
                X_FIXED.append(int(s))
            except ValueError:
                st.error(f"P{i+1} doit être un entier ou vide.")
                st.stop()

max_restarts = st.number_input("Nombre maximum de relances", value=2500, min_value=1, step=100)

# ------------------ Fonctions ------------------
def is_decreasing_positive(X):
    return all(x > 0 for x in X) and all(X[i] > X[i+1] for i in range(len(X)-1))

def simulate_once():
    # n dépend du choix de l'utilisateur
    n = len(X_FIXED)
    fixed_idx = [i for i, v in enumerate(X_FIXED) if v is not None]

    if not fixed_idx:  # Cas 1 : tout libre
        X = [0]*n
        left = target_sum
        for i in range(n-1):
            X[i] = random.randint(0, left)
            left -= X[i]
        X[-1] = left

        def value(X):
            return sum(w*x for w, x in zip(weights, X))

        def step(X):
            i, j = random.sample(range(n), 2)
            if X[j] == 0:
                return False
            X[i] += 1
            X[j] -= 1
            return True

    else:  # Cas 2 : certains fixés
        free_idx = [i for i in range(n) if i not in fixed_idx]
        sum_fixed = sum(X_FIXED[i] for i in fixed_idx)
        if sum_fixed > target_sum:
            return None, False, None, None  # invalide

        X = [0]*n
        for i in fixed_idx:
            X[i] = int(X_FIXED[i])
        left = target_sum - sum_fixed
        if free_idx:
            parts = [0]*len(free_idx)
            for k in range(len(free_idx)-1):
                take = random.randint(0, left)
                parts[k] = take
                left -= take
            parts[-1] = left
            random.shuffle(parts)
            for pos, i in enumerate(free_idx):
                X[i] = parts[pos]

        def value(X):
            return sum(w*x for w, x in zip(weights, X))

        def step(X):
            if len(free_idx) < 2:
                return False
            i, j = random.sample(free_idx, 2)
            if X[j] == 0:
                return False
            X[i] += 1
            X[j] -= 1
            return True

    # --- Recuit simulé (1000 itérations max) ---
    best = X[:]
    best_gap = abs(value(X) - target_val)
    T = 1.0
    for it in range(1000):
        gap = abs(value(X) - target_val)
        if gap == 0:
            break
        old = X[:]
        if not step(X):
            continue
        gap2 = abs(value(X) - target_val)
        if gap2 <= gap or random.random() < math.exp((gap - gap2) / max(1e-9, T)):
            if gap2 < best_gap:
                best, best_gap = X[:], gap2
        else:
            X[:] = old

        # Sécurités uniquement si des fixes existent
        if fixed_idx:
            for i in fixed_idx:
                X[i] = int(X_FIXED[i])
            s = sum(X)
            if s != target_sum and free_idx:
                X[free_idx[0]] += (target_sum - s)

    exact = (abs(value(X) - target_val) == 0)
    return X, exact, best, best_gap

def stats_for(sol_X):
    p = np.array(sol_X, dtype=float) / float(target_sum)
    # revenir aux multiplicateurs affichés (weights_display) pour stats
    vals = np.array(weights_display, dtype=float)  # déjà /10 par rapport aux weights
    mu = float(np.sum(p * vals))
    sigma = float(np.sqrt(np.sum(p * (vals - mu) ** 2)))
    return p, mu, sigma

# ------------------ Lancer (calculs uniquement au clic) ------------------
if st.button("Lancer la recherche"):
    solutions = []
    for r in range(1, int(max_restarts)+1):
        out = simulate_once()
        if out[0] is None:
            st.error("La somme des probabilités fixées dépasse 10000.")
            st.stop()
        X, exact, best, best_gap = out
        if exact and is_decreasing_positive(X):
            solutions.append((r, X, best_gap))
            if len(solutions) >= 10:  # Stop à 10 solutions exactes & monotones
                break

    if solutions:
        rows = []
        for idx, sol, gap in solutions:
            p, mu, sigma = stats_for(sol)
            rows.append({
                "P (valeurs)": sol,
                "Écart": gap,
                "Moyenne (M affichés)": mu,
                "Écart-type": sigma,
                "Probabilités (P/10000)": p.tolist()
            })
        # Tri par écart-type décroissant
        df = pd.DataFrame(rows).sort_values(by="Écart-type", ascending=False)
        st.subheader(
            f"Jusqu'à 10 solutions trouvées (exactes & monotones) par écart-type décroissant "
            f"(M1..M{n}, P1..P{n})"
        )
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("Aucune solution exacte trouvée. Relance la recherche ?")
