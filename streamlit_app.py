"""Streamlit entry point — run: streamlit run streamlit_app.py"""
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.config import load_config
from src.models.predict import predict_value
from src.utils.serialization import load_model

MODEL_PATH = "models/best_model.joblib"
CONFIG_PATH = "config/params.yaml"

_FR = {
    # Généraux
    "overall": "Note globale",
    "potential": "Potentiel",
    "potential_gap": "Écart potentiel",
    "age": "Âge",
    "height_cm": "Taille (cm)",
    "weight_kg": "Poids (kg)",
    "weak_foot": "Pied faible",
    "skill_moves": "Gestes techniques",
    "international_reputation": "Réputation internationale",
    # Ratings globaux
    "pace": "Vitesse",
    "shooting": "Tir",
    "passing": "Passe",
    "dribbling": "Dribble",
    "defending": "Défense",
    "physic": "Physique",
    # Attaque
    "attacking_crossing": "Centre",
    "attacking_finishing": "Finition",
    "attacking_heading_accuracy": "Jeu de tête",
    "attacking_short_passing": "Passe courte",
    "attacking_volleys": "Volée",
    # Technique
    "skill_dribbling": "Dribble (détail)",
    "skill_curve": "Effet",
    "skill_fk_accuracy": "Coup franc",
    "skill_long_passing": "Longue passe",
    "skill_ball_control": "Contrôle de balle",
    # Mouvement
    "movement_acceleration": "Accélération",
    "movement_sprint_speed": "Vitesse de pointe",
    "movement_agility": "Agilité",
    "movement_reactions": "Réactivité",
    "movement_balance": "Équilibre",
    # Puissance
    "power_shot_power": "Puissance de frappe",
    "power_jumping": "Détente",
    "power_stamina": "Endurance",
    "power_strength": "Force",
    "power_long_shots": "Frappe longue distance",
    # Mentalité
    "mentality_aggression": "Agressivité",
    "mentality_interceptions": "Interceptions",
    "mentality_positioning": "Placement",
    "mentality_vision": "Vision",
    "mentality_penalties": "Penaltys",
    "mentality_composure": "Sang-froid",
    # Défense
    "defending_marking_awareness": "Marquage",
    "defending_standing_tackle": "Tacle debout",
    "defending_sliding_tackle": "Tacle glissé",
    # Gardien
    "goalkeeping_diving": "Plongeon (GK)",
    "goalkeeping_handling": "Mains (GK)",
    "goalkeeping_kicking": "Relance (GK)",
    "goalkeeping_positioning": "Placement (GK)",
    "goalkeeping_reflexes": "Réflexes (GK)",
    "goalkeeping_speed": "Vitesse (GK)",
    # Catégorielles
    "preferred_foot_Left": "Pied gauche",
    "preferred_foot_Right": "Pied droit",
}

_WR_FR = {
    "High": "Haut", "Medium": "Moyen", "Low": "Bas",
}


def _translate(raw_name: str) -> str:
    """Convert a preprocessor feature name (e.g. 'num__age') to French."""
    name = raw_name.replace("num__", "").replace("cat__", "")
    if name in _FR:
        return _FR[name]
    # work_rate OHE: 'work_rate_High/Medium' → 'Pressing Haut/Moyen'
    if name.startswith("work_rate_"):
        wr = name.replace("work_rate_", "")
        parts = wr.split("/")
        fr_parts = [_WR_FR.get(p, p) for p in parts]
        return f"Pressing {'/'.join(fr_parts)}"
    return name.replace("_", " ").title()


@st.cache_resource
def _load_artifact():
    if not os.path.exists(MODEL_PATH):
        return None
    return load_model(MODEL_PATH)


@st.cache_data
def _load_config():
    return load_config(CONFIG_PATH)


def fmt_eur(value: float) -> str:
    if value >= 1_000_000:
        return f"€ {value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"€ {value / 1_000:.0f}K"
    return f"€ {value:,.0f}"


# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Football Scout — Valeur marchande",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ Outil de Scouting — Estimation de la valeur marchande")
st.caption("Simulez le profil d'un joueur et obtenez une estimation de sa valeur de transfert.")

# ── Load model ────────────────────────────────────────────────────────────────
artifact = _load_artifact()

if artifact is None:
    st.error(
        "Modèle introuvable. Lancez d'abord le pipeline : `python src/pipeline.py`"
    )
    st.stop()

config = _load_config()

# ── Sidebar — inputs ──────────────────────────────────────────────────────────
st.sidebar.header("Profil du joueur")

with st.sidebar:
    st.subheader("Informations générales")
    age = st.slider("Âge", 16, 42, 23)
    height_cm = st.slider("Taille (cm)", 155, 210, 181)
    weight_kg = st.slider("Poids (kg)", 50, 110, 75)
    preferred_foot = st.selectbox("Pied dominant", ["Right", "Left"])
    att_wr = st.selectbox("Pressing offensif", ["High", "Medium", "Low"], index=1)
    def_wr = st.selectbox("Pressing défensif", ["High", "Medium", "Low"], index=1)
    work_rate = f"{att_wr}/{def_wr}"
    weak_foot = st.slider("Pied faible (1–5)", 1, 5, 3)
    skill_moves = st.slider("Gestes techniques (1–5)", 1, 5, 3)
    international_reputation = st.slider("Réputation int. (1–5)", 1, 5, 1)

    st.subheader("Ratings globaux")
    overall = st.slider("Overall", 40, 99, 72)
    potential = st.slider("Potentiel", 40, 99, 78)
    pace = st.slider("Vitesse", 1, 99, 70)
    shooting = st.slider("Tir", 1, 99, 65)
    passing = st.slider("Passe", 1, 99, 68)
    dribbling = st.slider("Dribble", 1, 99, 70)
    defending = st.slider("Défense", 1, 99, 40)
    physic = st.slider("Physique", 1, 99, 72)

    with st.expander("Attributs offensifs détaillés"):
        attacking_crossing = st.slider("Centres", 1, 99, 65)
        attacking_finishing = st.slider("Finition", 1, 99, 65)
        attacking_heading_accuracy = st.slider("Jeu de tête", 1, 99, 60)
        attacking_short_passing = st.slider("Passe courte", 1, 99, 68)
        attacking_volleys = st.slider("Volée", 1, 99, 55)

    with st.expander("Attributs techniques"):
        skill_dribbling = st.slider("Dribble (détail)", 1, 99, 70)
        skill_curve = st.slider("Effet", 1, 99, 60)
        skill_fk_accuracy = st.slider("Précision coups francs", 1, 99, 55)
        skill_long_passing = st.slider("Longue passe", 1, 99, 62)
        skill_ball_control = st.slider("Contrôle de balle", 1, 99, 70)

    with st.expander("Attributs physiques / mouvement"):
        movement_acceleration = st.slider("Accélération", 1, 99, 72)
        movement_sprint_speed = st.slider("Vitesse de pointe", 1, 99, 70)
        movement_agility = st.slider("Agilité", 1, 99, 68)
        movement_reactions = st.slider("Réactivité", 1, 99, 70)
        movement_balance = st.slider("Équilibre", 1, 99, 68)
        power_shot_power = st.slider("Puissance de frappe", 1, 99, 72)
        power_jumping = st.slider("Détente", 1, 99, 65)
        power_stamina = st.slider("Endurance", 1, 99, 74)
        power_strength = st.slider("Force", 1, 99, 68)
        power_long_shots = st.slider("Frappe longue distance", 1, 99, 62)

    with st.expander("Attributs mentaux"):
        mentality_aggression = st.slider("Agressivité", 1, 99, 65)
        mentality_interceptions = st.slider("Interceptions", 1, 99, 45)
        mentality_positioning = st.slider("Placement", 1, 99, 68)
        mentality_vision = st.slider("Vision", 1, 99, 68)
        mentality_penalties = st.slider("Penalties", 1, 99, 60)
        mentality_composure = st.slider("Sang-froid", 1, 99, 70)

    with st.expander("Attributs défensifs"):
        defending_marking_awareness = st.slider("Marquage", 1, 99, 40)
        defending_standing_tackle = st.slider("Tacle debout", 1, 99, 40)
        defending_sliding_tackle = st.slider("Tacle glissé", 1, 99, 38)

    goalkeeper_mode = st.checkbox("Profil gardien de but")
    if goalkeeper_mode:
        with st.expander("Attributs gardien"):
            goalkeeping_diving = st.slider("Plongeon", 1, 99, 70)
            goalkeeping_handling = st.slider("Mains", 1, 99, 68)
            goalkeeping_kicking = st.slider("Relance", 1, 99, 60)
            goalkeeping_positioning = st.slider("Placement (GK)", 1, 99, 68)
            goalkeeping_reflexes = st.slider("Réflexes", 1, 99, 72)
            goalkeeping_speed = st.slider("Vitesse (GK)", 1, 99, 55)
    else:
        goalkeeping_diving = goalkeeping_handling = goalkeeping_kicking = 0
        goalkeeping_positioning = goalkeeping_reflexes = goalkeeping_speed = 0

# ── Build feature dict ────────────────────────────────────────────────────────
player_features = dict(
    age=age, height_cm=height_cm, weight_kg=weight_kg,
    preferred_foot=preferred_foot, work_rate=work_rate,
    weak_foot=weak_foot, skill_moves=skill_moves,
    international_reputation=international_reputation,
    overall=overall, potential=potential,
    pace=pace, shooting=shooting, passing=passing,
    dribbling=dribbling, defending=defending, physic=physic,
    attacking_crossing=attacking_crossing,
    attacking_finishing=attacking_finishing,
    attacking_heading_accuracy=attacking_heading_accuracy,
    attacking_short_passing=attacking_short_passing,
    attacking_volleys=attacking_volleys,
    skill_dribbling=skill_dribbling, skill_curve=skill_curve,
    skill_fk_accuracy=skill_fk_accuracy, skill_long_passing=skill_long_passing,
    skill_ball_control=skill_ball_control,
    movement_acceleration=movement_acceleration,
    movement_sprint_speed=movement_sprint_speed,
    movement_agility=movement_agility, movement_reactions=movement_reactions,
    movement_balance=movement_balance,
    power_shot_power=power_shot_power, power_jumping=power_jumping,
    power_stamina=power_stamina, power_strength=power_strength,
    power_long_shots=power_long_shots,
    mentality_aggression=mentality_aggression,
    mentality_interceptions=mentality_interceptions,
    mentality_positioning=mentality_positioning,
    mentality_vision=mentality_vision,
    mentality_penalties=mentality_penalties,
    mentality_composure=mentality_composure,
    defending_marking_awareness=defending_marking_awareness,
    defending_standing_tackle=defending_standing_tackle,
    defending_sliding_tackle=defending_sliding_tackle,
    goalkeeping_diving=goalkeeping_diving,
    goalkeeping_handling=goalkeeping_handling,
    goalkeeping_kicking=goalkeeping_kicking,
    goalkeeping_positioning=goalkeeping_positioning,
    goalkeeping_reflexes=goalkeeping_reflexes,
    goalkeeping_speed=goalkeeping_speed,
    # engineered feature
    potential_gap=potential - overall,
)

# ── Main panel ────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    estimate_btn = st.button("🔍 Estimer la valeur", type="primary", use_container_width=True)

st.divider()

if estimate_btn:
    with st.spinner("Calcul en cours…"):
        estimated_value = predict_value(artifact, player_features)

    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_b:
        st.metric(
            label=f"Valeur marchande estimée — {artifact['best_model_name'].replace('_', ' ').title()}",
            value=fmt_eur(estimated_value),
        )
        gap = potential - overall
        if gap >= 10:
            st.success(f"📈 Profil sous-coté — potentiel_gap = +{gap} pts (cible scouting idéale)")
        elif gap >= 5:
            st.info(f"📊 Profil en développement — potentiel_gap = +{gap} pts")
        else:
            st.warning(f"⚠️ Joueur proche de son pic — potentiel_gap = {gap} pts")

    # SHAP waterfall for this player
    with st.expander("🔬 Facteurs d'influence (SHAP)", expanded=False):
        try:
            from src.models.explain import explain_single_player

            player_df = pd.DataFrame([{
                k: v for k, v in player_features.items()
                if k in artifact["numerical_features"] + artifact["categorical_features"]
            }])
            sv, feat_names = explain_single_player(
                artifact["pipeline"],
                player_df,
                X_background=artifact.get("X_background"),
            )

            # Top-15 features by absolute SHAP value
            pairs = sorted(zip(feat_names, sv), key=lambda x: abs(x[1]), reverse=True)[:15]
            names_top = [_translate(p[0]) for p in pairs]
            vals_top = [p[1] for p in pairs]
            colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in vals_top]

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(names_top[::-1], vals_top[::-1], color=colors[::-1])
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Impact SHAP (échelle log de la valeur)")
            ax.set_title("Top 15 facteurs d'influence sur l'estimation")
            ax.tick_params(axis="y", labelsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Légende
            st.caption(
                "🟢 Augmente la valeur estimée  —  🔴 Diminue la valeur estimée"
            )
        except Exception as exc:
            st.info(f"SHAP non disponible pour ce modèle : {exc}")

else:
    st.info("👈 Ajustez les caractéristiques du joueur dans la barre latérale puis cliquez sur **Estimer la valeur**.")

    # Quick summary stats
    st.subheader("Résumé du profil saisi")
    summary_df = pd.DataFrame(
        {
            "Attribut": ["Overall", "Potentiel", "Potentiel gap", "Âge", "Vitesse", "Tir", "Passe", "Dribble", "Défense", "Physique"],
            "Valeur": [overall, potential, f"+{potential-overall}", age, pace, shooting, passing, dribbling, defending, physic],
        }
    )
    st.table(summary_df.set_index("Attribut"))
