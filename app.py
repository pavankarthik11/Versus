import os
import random
import traceback

import numpy as np
import pandas as pd
import joblib
import streamlit as st


# ================================================================
# PAGE CONFIG
# ================================================================

st.set_page_config(
    page_title="VERSUS - IPL Custom XI",
    page_icon="🏏",
    layout="wide"
)


# ================================================================
# STYLE
# ================================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 44px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #777;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .score {
        font-size: 42px;
        font-weight: 800;
    }

    .winner {
        font-size: 30px;
        font-weight: 800;
        text-align: center;
        padding: 25px;
    }

    .auction-player {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        padding: 25px;
    }

    .bid-price {
        text-align: center;
        font-size: 34px;
        font-weight: 800;
        padding: 10px;
    }

    .team-card {
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #ddd;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ================================================================
# PROJECT DIRECTORY
# ================================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ================================================================
# FIND FILE
# ================================================================

def find_file(filename):

    for root, dirs, files in os.walk(BASE_DIR):

        if filename in files:

            return os.path.join(
                root,
                filename
            )

    return None


# ================================================================
# LOAD ML ENGINE
# ================================================================

try:

    from custom_xi_engine import (
        simulate_innings,
        simulate_match,
        find_bowlers
    )

except Exception:

    st.error(
        "❌ Could not load custom_xi_engine.py"
    )

    st.code(
        traceback.format_exc()
    )

    st.stop()


# ================================================================
# LOAD PLAYER DATA
# ================================================================

@st.cache_resource
def load_player_data():

    player_profile_path = find_file(
        "player_profile.pkl"
    )

    matchup_path = find_file(
        "matchup_df.pkl"
    )


    if player_profile_path is None:

        raise FileNotFoundError(
            "player_profile.pkl not found."
        )


    if matchup_path is None:

        raise FileNotFoundError(
            "matchup_df.pkl not found."
        )


    player_profile = pd.read_pickle(
        player_profile_path
    )

    matchup_df = pd.read_pickle(
        matchup_path
    )


    return (
        player_profile,
        matchup_df
    )


try:

    (
        player_profile,
        matchup_df
    ) = load_player_data()

except Exception:

    st.error(
        "❌ Could not load player data."
    )

    st.code(
        traceback.format_exc()
    )

    st.stop()


# ================================================================
# NORMALIZE PLAYER DATA
# ================================================================

if "player" not in player_profile.columns:

    st.error(
        "player_profile.pkl does not contain a 'player' column."
    )

    st.stop()


if "batter" not in matchup_df.columns:

    st.error(
        "matchup_df.pkl does not contain a 'batter' column."
    )

    st.stop()


player_profile["player"] = (
    player_profile["player"]
    .astype(str)
    .str.strip()
)


matchup_df["batter"] = (
    matchup_df["batter"]
    .astype(str)
    .str.strip()
)


if "bowler" in matchup_df.columns:

    matchup_df["bowler"] = (
        matchup_df["bowler"]
        .astype(str)
        .str.strip()
    )


# ================================================================
# PLAYER LIST
# ================================================================

all_players = sorted(
    player_profile[
        "player"
    ]
    .dropna()
    .unique()
    .tolist()
)


# ================================================================
# ROLE MAP
# ================================================================

role_map = {}

for _, row in player_profile.iterrows():

    player = str(
        row["player"]
    ).strip()

    role_map[player] = str(
        row.get(
            "role",
            ""
        )
    ).lower().strip()


# ================================================================
# RANDOM 30-PLAYER AUCTION POOL
# ================================================================
#
# This is an ADDITION to the existing manual selection.
# The normal manual "Choose 30 players" option remains unchanged.
#
# Random mode:
#   - uses the most established IPL players by matches played
#   - considers the top 100 in each role when available
#   - keeps a healthy mix of batsmen, all-rounders and bowlers
#   - randomly samples from those popular pools, so the result changes
#     when the button is pressed
#
# Target composition for the 30-player auction pool:
#   12 batsmen / wicketkeepers
#    6 all-rounders
#   12 bowlers
#
# The role/match columns are detected defensively because different
# versions of player_profile.pkl may use different column names.
# ================================================================

MATCH_COLUMNS = [
    "matches",
    "matches_played",
    "total_matches",
    "match_count",
    "games",
    "appearances",
    "ipl_matches",
    "num_matches",
]


def _numeric_column(df, candidates):
    """Return the first useful numeric column from candidates."""
    for column in candidates:
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce")
            if values.notna().any():
                return column
    return None


def _build_match_count_map():
    """Build IPL appearance counts without changing the existing data."""
    counts = {}

    # Prefer an explicit matches column in player_profile.pkl.
    explicit_col = _numeric_column(player_profile, MATCH_COLUMNS)
    if explicit_col is not None:
        for _, row in player_profile.iterrows():
            player = str(row.get("player", "")).strip()
            value = pd.to_numeric(row.get(explicit_col), errors="coerce")
            if player and pd.notna(value):
                counts[player] = max(counts.get(player, 0), int(value))

    # If no explicit count exists, derive an appearance proxy from
    # matchup_df. Prefer unique match_id, then date, then row count.
    if "match_id" in matchup_df.columns:
        for player_column in ["batter", "bowler"]:
            if player_column not in matchup_df.columns:
                continue
            temp = matchup_df[[player_column, "match_id"]].copy()
            temp[player_column] = temp[player_column].astype(str).str.strip()
            temp = temp.dropna(subset=[player_column, "match_id"])
            grouped = temp.groupby(player_column)["match_id"].nunique()
            for player, value in grouped.items():
                counts[player] = max(counts.get(player, 0), int(value))

    elif "date" in matchup_df.columns:
        for player_column in ["batter", "bowler"]:
            if player_column not in matchup_df.columns:
                continue
            temp = matchup_df[[player_column, "date"]].copy()
            temp[player_column] = temp[player_column].astype(str).str.strip()
            temp["date"] = pd.to_datetime(temp["date"], errors="coerce")
            temp = temp.dropna(subset=[player_column, "date"])
            grouped = temp.groupby(player_column)["date"].nunique()
            for player, value in grouped.items():
                counts[player] = max(counts.get(player, 0), int(value))

    # Final fallback: number of historical rows involving the player.
    for player_column in ["batter", "bowler"]:
        if player_column not in matchup_df.columns:
            continue
        series = matchup_df[player_column].astype(str).str.strip()
        grouped = series.value_counts()
        for player, value in grouped.items():
            counts[player] = max(counts.get(player, 0), int(value))

    return counts


def _player_role(player):
    """Normalize the role into batsman / allrounder / bowler."""
    role = role_map.get(player, "")
    role = str(role).lower().replace("–", "-").strip()

    if (
        "all-round" in role
        or "all round" in role
        or "allround" in role
    ):
        return "allrounder"

    if (
        "bowler" in role
        or "bowling" in role
        or role in {"bowl", "bowling all-rounder"}
    ):
        return "bowler"

    if (
        "batter" in role
        or "batsman" in role
        or "wicketkeeper" in role
        or "wicket keeper" in role
        or "keeper" in role
    ):
        return "batsman"

    # Unknown roles are left out of the specialist pools initially.
    return "unknown"


def get_random_balanced_auction_players(total=30):
    """
    Build EXACTLY 30 random auction players from three strict Top-50 pools:

        12 batsmen / wicketkeepers
         6 all-rounders
        12 bowlers

    Ranking inside each role is based on IPL matches played.
    Randomness happens only AFTER the Top-50 list for each role is built.
    """

    total = int(total)

    if total != 30:
        raise ValueError(
            "Random auction pool must contain exactly 30 players."
        )

    match_counts = _build_match_count_map()

    rows = []

    for player in all_players:
        rows.append({
            "player": player,
            "role": _player_role(player),
            "matches": int(match_counts.get(player, 0)),
        })

    stats = pd.DataFrame(rows)

    # ------------------------------------------------------------
    # STRICT TOP 50 POOLS
    # ------------------------------------------------------------
    #
    # IMPORTANT:
    # Ranking is by IPL matches played.
    # Wicketkeepers are part of the batsman pool.
    # ------------------------------------------------------------

    role_pools = {}

    for role in [
        "batsman",
        "allrounder",
        "bowler"
    ]:

        role_df = stats[
            stats["role"] == role
        ].copy()

        role_df = role_df.sort_values(
            ["matches", "player"],
            ascending=[False, True]
        ).head(50)

        role_pools[role] = role_df[
            "player"
        ].tolist()

    # ------------------------------------------------------------
    # EXACT COMPOSITION
    # ------------------------------------------------------------

    targets = {
        "batsman": 12,
        "allrounder": 6,
        "bowler": 12,
    }

    # Do NOT silently change the requested composition.
    # If the dataset cannot provide one of the categories, show a
    # clear error instead of producing an incorrect 30-player pool.

    shortages = {}

    for role, required in targets.items():
        available = len(role_pools[role])

        if available < required:
            shortages[role] = {
                "required": required,
                "available": available
            }

    if shortages:
        details = "; ".join(
            f"{role}: need {info['required']}, found {info['available']}"
            for role, info in shortages.items()
        )

        raise ValueError(
            "Not enough classified players for the exact "
            f"12 batsmen/WKs + 6 all-rounders + 12 bowlers requirement. {details}"
        )

    selected = []

    # Randomly choose from each STRICT Top-50 pool.
    selected.extend(
        random.sample(
            role_pools["batsman"],
            targets["batsman"]
        )
    )

    selected.extend(
        random.sample(
            role_pools["allrounder"],
            targets["allrounder"]
        )
    )

    selected.extend(
        random.sample(
            role_pools["bowler"],
            targets["bowler"]
        )
    )

    # Final shuffle so auction order is not role-by-role.
    random.shuffle(selected)

    if len(selected) != 30 or len(set(selected)) != 30:
        raise ValueError(
            "Internal error: automatic auction pool is not exactly 30 unique players."
        )

    return selected


# ================================================================
# SESSION STATE INITIALIZATION
# ================================================================

DEFAULT_STATE = {

    "page":
        "home",

    "prediction":
        None,

    "auction_started":
        False,

    "auction_finished":
        False,

    "auction_players":
        [],

    "random_auction_pool":
        [],

    "auction_index":
        0,

    "current_player":
        None,

    "current_bid":
        0.0,

    "highest_bidder":
        None,

    "team1_purse":
        120.0,

    "team2_purse":
        120.0,

    "team1_squad":
        [],

    "team2_squad":
        [],

    "team1_stopped":
        False,

    "team2_stopped":
        False,

    "auction_history":
        [],

    "auction_message":
        "",

    "auction_prediction_started":
        False,

    "auction_team1_bowlers":
        [],

    "auction_team2_bowlers":
        [],

    # Playing XI selected from the 15 players bought at auction.
    # The order in these lists is the batting order used by prediction.
    "auction_playing11_team1":
        [],
    "auction_playing11_team2":
        [],

    "simulation_seed":
        42,

    "custom_toss_call_team1": "Heads",
    "custom_toss_call_team2": "Tails",
    "custom_toss_winner": None,
    "custom_toss_decision": None,
    "custom_toss_signature": None,

    "auction_toss_call_team1": "Heads",
    "auction_toss_call_team2": "Tails",
    "auction_toss_winner": None,
    "auction_toss_decision": None,
    "auction_toss_signature": None
}


for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ================================================================
# RESET AUCTION
# ================================================================

def reset_auction():

    st.session_state.auction_started = False

    st.session_state.auction_finished = False

    st.session_state.auction_players = []

    st.session_state.random_auction_pool = []

    st.session_state.auction_index = 0

    st.session_state.current_player = None

    st.session_state.current_bid = 0.0

    st.session_state.highest_bidder = None

    st.session_state.team1_purse = 120.0

    st.session_state.team2_purse = 120.0

    st.session_state.team1_squad = []

    st.session_state.team2_squad = []

    st.session_state.team1_stopped = False

    st.session_state.team2_stopped = False

    st.session_state.auction_history = []

    st.session_state.auction_message = ""

    st.session_state.auction_prediction_started = False

    st.session_state.auction_team1_bowlers = []

    st.session_state.auction_team2_bowlers = []

    st.session_state.auction_playing11_team1 = []
    st.session_state.auction_playing11_team2 = []

    st.session_state.prediction = None


# ================================================================
# START AUCTION
# ================================================================

def start_auction(players):

    players = list(players)

    random.shuffle(
        players
    )

    st.session_state.auction_players = players

    st.session_state.auction_index = 0

    st.session_state.current_player = None

    st.session_state.current_bid = 0.0

    st.session_state.highest_bidder = None

    st.session_state.team1_purse = 120.0

    st.session_state.team2_purse = 120.0

    st.session_state.team1_squad = []

    st.session_state.team2_squad = []

    st.session_state.auction_history = []

    st.session_state.team1_stopped = False

    st.session_state.team2_stopped = False

    st.session_state.auction_message = ""

    st.session_state.auction_finished = False

    st.session_state.auction_started = True

    st.session_state.auction_prediction_started = False


# ================================================================
# SPIN WHEEL
# ================================================================

def spin_wheel():

    if not st.session_state.auction_started:

        return


    if st.session_state.auction_finished:

        return


    # ------------------------------------------------------------
    # Auction ends only when BOTH teams have 15.
    # ------------------------------------------------------------

    if (
        len(st.session_state.team1_squad) >= 11
        and
        len(st.session_state.team2_squad) >= 11
    ):
        st.session_state.auction_finished = True
        return

    # ------------------------------------------------------------
    # Get next player from the randomized auction order.
    # ------------------------------------------------------------

    players = st.session_state.auction_players
    index = st.session_state.auction_index

    if index >= len(players):
        st.session_state.auction_finished = True
        return

    player = players[index]
    st.session_state.auction_index += 1

    team1_count = len(st.session_state.team1_squad)
    team2_count = len(st.session_state.team2_squad)

    # ------------------------------------------------------------
    # If Team 1 is already full, Team 2 automatically receives
    # every remaining player at the ₹1 Cr base price.
    # No bidding is shown.
    # ------------------------------------------------------------

    if team1_count >= 15 and team2_count < 15:
        st.session_state.current_player = player
        st.session_state.current_bid = 1.0
        st.session_state.highest_bidder = "TEAM 2"

        complete_player("TEAM 2")
        return

    # ------------------------------------------------------------
    # If Team 2 is already full, Team 1 automatically receives
    # every remaining player at the ₹1 Cr base price.
    # No bidding is shown.
    # ------------------------------------------------------------

    if team2_count >= 15 and team1_count < 15:
        st.session_state.current_player = player
        st.session_state.current_bid = 1.0
        st.session_state.highest_bidder = "TEAM 1"

        complete_player("TEAM 1")
        return

    # ------------------------------------------------------------
    # Normal auction while both teams still have room.
    # ------------------------------------------------------------

    st.session_state.current_player = player
    st.session_state.current_bid = 0.0
    st.session_state.highest_bidder = None
    st.session_state.team1_stopped = False
    st.session_state.team2_stopped = False

    st.session_state.auction_message = (
        f"🎡 The wheel selected {player}"
    )


# ================================================================
# BID
# ================================================================

def place_bid(
    team,
    increment
):

    # ============================================================
    # CHECK TEAM SIZE
    # ============================================================

    if team == "TEAM 1":

        if len(
            st.session_state.team1_squad
        ) >= 15:

            st.warning(
                "Team 1 already has 15 players."
            )

            return


        purse = st.session_state.team1_purse


    else:

        if len(
            st.session_state.team2_squad
        ) >= 15:

            st.warning(
                "Team 2 already has 15 players."
            )

            return


        purse = st.session_state.team2_purse


    # ============================================================
    # CHECK IF TEAM STOPPED
    # ============================================================

    if team == "TEAM 1":

        if st.session_state.team1_stopped:

            return

    else:

        if st.session_state.team2_stopped:

            return


    # ============================================================
    # NEW BID
    # ============================================================

    new_bid = (
        st.session_state.current_bid
        +
        float(increment)
    )


    if new_bid > purse:

        st.warning(
            f"{team} does not have enough purse."
        )

        return


    st.session_state.current_bid = new_bid

    st.session_state.highest_bidder = team

    st.session_state.auction_message = (
        f"🔨 {team} bid ₹{new_bid:.2f} Cr"
    )


    # ============================================================
    # RESET OPPONENT STOP
    # ============================================================

    if team == "TEAM 1":

        st.session_state.team2_stopped = False

    else:

        st.session_state.team1_stopped = False


# ================================================================
# STOP BIDDING
# ================================================================

def stop_bidding(team):
    # Team has stopped bidding on the current player.
    # If the other team is already highest bidder, sell immediately.

    if team == "TEAM 1":
        if len(st.session_state.team1_squad) >= 15:
            return
        st.session_state.team1_stopped = True
        other_team = "TEAM 2"
        other_has_space = len(st.session_state.team2_squad) < 15
        other_stopped = st.session_state.team2_stopped
    else:
        if len(st.session_state.team2_squad) >= 15:
            return
        st.session_state.team2_stopped = True
        other_team = "TEAM 1"
        other_has_space = len(st.session_state.team1_squad) < 15
        other_stopped = st.session_state.team1_stopped

    highest = st.session_state.highest_bidder

    # If the opponent has the highest bid, this STOP ends the auction.
    if highest == other_team and other_has_space:
        complete_player(other_team)
        return

    # If this team is highest bidder and the opponent has already stopped,
    # this team wins the player immediately.
    if highest == team and other_stopped:
        complete_player(team)
        return

    # No bid yet: give the other team a chance.
    if highest is None:
        if other_has_space:
            st.session_state.auction_message = (
                f"🛑 {team} stopped bidding. {other_team} can bid."
            )
            return
        pass_player()
        return

    # Current team stopped while opponent can still bid.
    st.session_state.auction_message = (
        f"🛑 {team} stopped bidding. {other_team} can bid."
    )

# ================================================================
# COMPLETE PLAYER
# ================================================================

def complete_player(team):

    player = (
        st.session_state.current_player
    )


    if player is None:

        return


    price = float(
        st.session_state.current_bid
    )


    # ============================================================
    # FIRST PLAYER BASE PRICE
    # ============================================================

    if price <= 0:

        price = 1.0


    # ============================================================
    # TEAM 1
    # ============================================================

    if team == "TEAM 1":

        if len(
            st.session_state.team1_squad
        ) >= 15:

            st.warning(
                "Team 1 already has 15 players."
            )

            return


        if price > st.session_state.team1_purse:

            st.error(
                "Team 1 cannot afford this player."
            )

            return


        st.session_state.team1_squad.append(
            player
        )

        st.session_state.team1_purse -= price


    # ============================================================
    # TEAM 2
    # ============================================================

    else:

        if len(
            st.session_state.team2_squad
        ) >= 15:

            st.warning(
                "Team 2 already has 15 players."
            )

            return


        if price > st.session_state.team2_purse:

            st.error(
                "Team 2 cannot afford this player."
            )

            return


        st.session_state.team2_squad.append(
            player
        )

        st.session_state.team2_purse -= price


    # ============================================================
    # AUCTION HISTORY
    # ============================================================

    st.session_state.auction_history.append({

        "Player":
            player,

        "Team":
            team,

        "Price":
            round(
                price,
                2
            )
    })


    st.session_state.auction_message = (
        f"🎉 {player} SOLD to {team} "
        f"for ₹{price:.2f} Cr"
    )


    # ============================================================
    # CLEAR CURRENT PLAYER
    # ============================================================

    st.session_state.current_player = None

    st.session_state.current_bid = 0.0

    st.session_state.highest_bidder = None

    st.session_state.team1_stopped = False

    st.session_state.team2_stopped = False


    # ============================================================
    # EXACTLY 15 + 15
    # ============================================================

    if (
        len(st.session_state.team1_squad)
        ==
        15
        and
        len(st.session_state.team2_squad)
        ==
        15
    ):

        st.session_state.auction_finished = True

        st.session_state.auction_message = (
            "🎉 AUCTION COMPLETE — "
            "Both teams have exactly 15 players."
        )


# ================================================================
# PASS PLAYER
# ================================================================

def pass_player():

    player = (
        st.session_state.current_player
    )


    if player is None:

        return


    st.session_state.auction_message = (
        f"⏭️ {player} was passed."
    )


    st.session_state.current_player = None

    st.session_state.current_bid = 0.0

    st.session_state.highest_bidder = None

    st.session_state.team1_stopped = False

    st.session_state.team2_stopped = False


# ================================================================
# AUTO-HANDLE WHEN ONE TEAM REACHES 15
# ================================================================

def auto_finish_if_needed():

    team1_count = len(
        st.session_state.team1_squad
    )

    team2_count = len(
        st.session_state.team2_squad
    )


    # ============================================================
    # TEAM 1 FULL
    # ============================================================

    if team1_count >= 15:

        # If current player exists and Team 2 can receive
        # him, Team 2 gets the player at current bid.

        if (
            st.session_state.current_player
            is not None
            and
            team2_count < 15
        ):

            if (
                st.session_state.current_bid
                <=
                st.session_state.team2_purse
            ):

                complete_player(
                    "TEAM 2"
                )

                return


    # ============================================================
    # TEAM 2 FULL
    # ============================================================

    if team2_count >= 15:

        if (
            st.session_state.current_player
            is not None
            and
            team1_count < 15
        ):

            if (
                st.session_state.current_bid
                <=
                st.session_state.team1_purse
            ):

                complete_player(
                    "TEAM 1"
                )

                return


# ================================================================
# FIND VALID BOWLERS
# ================================================================

def safe_find_bowlers(team):

    try:

        result = find_bowlers(
            team
        )

        if result is None:

            return []

        return list(
            dict.fromkeys(
                result
            )
        )

    except Exception:

        return []


# ================================================================
# HOME PAGE
# ================================================================

def home_page():

    st.markdown(
        '<div class="main-title">🏏 VERSUS</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'AI-Powered IPL Custom XI & Auction Simulator'
        '</div>',
        unsafe_allow_html=True
    )


    st.divider()


    st.header(
        "🏠 Choose Your Mode"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.subheader(
            "⚔️ Custom XI"
        )

        st.write(
            "Directly create two custom teams "
            "and simulate the match."
        )


        if st.button(
            "Create Custom XI",
            use_container_width=True
        ):

            st.session_state.page = "custom"

            st.rerun()


    with col2:

        st.subheader(
            "🔨 Player Auction"
        )

        st.write(
            "Select 30 players and conduct a "
            "live auction with ₹120 Cr purse."
        )


        if st.button(
            "Start Auction",
            use_container_width=True
        ):

            reset_auction()

            st.session_state.page = "auction"

            st.rerun()


    st.divider()


    st.subheader(
        "🤖 ML Engine"
    )


    st.write(
        f"Historical matchup records: "
        f"**{len(matchup_df):,}**"
    )

    st.write(
        f"Available players: "
        f"**{len(all_players):,}**"
    )


# ================================================================
# TOSS UI / LOGIC
# ================================================================

def toss_section(prefix, team1_label, team2_label, signature):
    signature_key = f"{prefix}_toss_signature"
    winner_key = f"{prefix}_toss_winner"
    decision_key = f"{prefix}_toss_decision"

    if st.session_state.get(signature_key) != signature:
        st.session_state[signature_key] = signature
        st.session_state[winner_key] = None
        st.session_state[decision_key] = None

    st.divider()
    st.header("🪙 Toss")
    st.caption("Choose Heads or Tails for both teams, then flip the coin.")

    col1, col2 = st.columns(2)

    with col1:
        call1 = st.selectbox(
            f"{team1_label} call",
            ["Heads", "Tails"],
            key=f"{prefix}_toss_call_team1"
        )

    with col2:
        call2 = st.selectbox(
            f"{team2_label} call",
            ["Heads", "Tails"],
            key=f"{prefix}_toss_call_team2"
        )

    if call1 == call2:
        st.warning(
            "Both teams cannot call the same side. One team must choose Heads and the other Tails."
        )
        st.session_state[winner_key] = None
        st.session_state[decision_key] = None
        return None, None

    if st.session_state.get(winner_key) is None:
        if st.button(
            "🪙 FLIP TOSS",
            type="primary",
            use_container_width=True,
            key=f"{prefix}_flip_toss"
        ):
            coin = "Heads" if np.random.random() < 0.5 else "Tails"
            winner = team1_label if coin == call1 else team2_label
            st.session_state[winner_key] = winner
            st.session_state[decision_key] = None
            st.rerun()

    winner = st.session_state.get(winner_key)

    if winner:
        coin = call1 if winner == team1_label else call2
        st.success(f"🪙 Toss: **{coin}** — **{winner} won the toss!**")

        decision = st.radio(
            f"{winner} — choose",
            ["BAT", "BOWL"],
            horizontal=True,
            key=f"{prefix}_toss_decision_radio"
        )

        st.session_state[decision_key] = decision
        return winner, decision

    return None, None


# ================================================================
# CUSTOM XI PAGE
# ================================================================

def custom_page():

    if st.button(
        "⬅️ Back Home"
    ):

        st.session_state.page = "home"

        st.rerun()


    st.markdown(
        '<div class="main-title">'
        '🏏 Custom XI Match'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="subtitle">'
        'Build two teams and simulate the match using ML.'
        '</div>',
        unsafe_allow_html=True
    )


    # ============================================================
    # TEAM SELECTION
    # ============================================================

    st.header(
        "🏟️ Select Your Teams"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.subheader(
            "🔵 TEAM 1"
        )

        team1 = st.multiselect(

            "Team 1 Players",

            all_players,

            max_selections=11,

            key="custom_team1"
        )

        st.info(
            f"Team 1: {len(team1)}/11"
        )


    with col2:

        st.subheader(
            "🟠 TEAM 2"
        )

        team2 = st.multiselect(

            "Team 2 Players",

            all_players,

            max_selections=11,

            key="custom_team2"
        )

        st.info(
            f"Team 2: {len(team2)}/11"
        )


    duplicates = set(
        team1
    ).intersection(
        team2
    )


    if duplicates:

        st.error(
            "Same player cannot be in both teams: "
            +
            ", ".join(
                sorted(
                    duplicates
                )
            )
        )


    # ============================================================
    # BOWLING
    # ============================================================

    st.header(
        "🎯 Select Bowling Attacks"
    )


    st.caption(
        "Choose up to 5 bowlers from each selected XI."
    )


    team1_possible_bowlers = safe_find_bowlers(
        team1
    )

    team2_possible_bowlers = safe_find_bowlers(
        team2
    )


    col1, col2 = st.columns(2)


    with col1:

        team1_bowlers = st.multiselect(

            "Team 1 Bowling Attack",

            team1_possible_bowlers,

            max_selections=5,

            key="custom_team1_bowlers"
        )


    with col2:

        team2_bowlers = st.multiselect(

            "Team 2 Bowling Attack",

            team2_possible_bowlers,

            max_selections=5,

            key="custom_team2_bowlers"
        )

    # ============================================================
    # TOSS — AFTER PLAYING XI / BOWLING SELECTION
    # ============================================================

    custom_toss_signature = (
        tuple(team1),
        tuple(team2),
        tuple(team1_bowlers),
        tuple(team2_bowlers)
    )

    toss_winner, toss_decision = toss_section(
        "custom",
        "TEAM 1",
        "TEAM 2",
        custom_toss_signature
    )



    # ============================================================
    # SETTINGS
    # ============================================================

    st.header(
        "⚙️ Match Settings"
    )


    col1, col2 = st.columns(2)


    with col1:

        overs = st.number_input(

            "Overs",

            min_value=1,

            max_value=20,

            value=20,

            step=1,

            key="custom_overs"
        )


    with col2:

        st.info(
            "🎲 Every prediction uses fresh random simulation. "
            "The same teams can produce a different result each time."
        )


    # ============================================================
    # PREDICT
    # ============================================================

    if st.button(
        "🏏 PREDICT MATCH",
        type="primary",
        use_container_width=True
    ):

        if len(team1) != 11:

            st.error(
                "Team 1 must contain exactly 11 players."
            )

            return


        if len(team2) != 11:

            st.error(
                "Team 2 must contain exactly 11 players."
            )

            return


        if duplicates:

            st.error(
                "Remove duplicate players."
            )

            return


        if len(team1_bowlers) < 5:

            st.error(
                "Select at least 5 Team 1 bowlers."
            )

            return


        if len(team2_bowlers) < 5:

            st.error(
                "Select at least 5 Team 2 bowlers."
            )

            return

        if toss_winner is None or toss_decision is None:
            st.error(
                "Complete the toss and let the toss winner choose BAT or BOWL before predicting."
            )
            return


        # IMPORTANT: never reuse a fixed seed here.
        # Every press of PREDICT MATCH starts a fresh stochastic simulation.
        np.random.seed(None)


        with st.spinner(
            "🤖 Running ML simulation..."
        ):

            try:

                match_result = simulate_match(
                    team1,
                    team1_bowlers,
                    team2,
                    team2_bowlers,
                    int(overs),
                    toss_winner=toss_winner,
                    toss_decision=toss_decision
                )

                result1 = match_result["team1"]
                result2 = match_result["team2"]


            except Exception:

                st.error(
                    "❌ Prediction failed."
                )

                st.code(
                    traceback.format_exc()
                )

                return


        display_match_result(
            result1,
            result2,
            "TEAM 1",
            "TEAM 2",
            match_result
        )


# ================================================================
# DISPLAY MATCH RESULT
# ================================================================

def display_match_result(
    result1,
    result2,
    name1,
    name2,
    match_result=None
):

    st.divider()

    st.header("🏆 Predicted Match Result")

    # ------------------------------------------------------------
    # TOSS / INNINGS INFORMATION
    # ------------------------------------------------------------
    if match_result:
        toss_winner = match_result.get("toss_winner")
        toss_decision = match_result.get("toss_decision")
        first_batting = match_result.get("first_batting")

        if toss_winner and toss_decision and first_batting:
            st.info(
                f"🪙 Toss: **{toss_winner} won** and chose **{toss_decision}**. "
                f"**{first_batting}** bats first."
            )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"🔵 {name1}")
        st.markdown(
            f'<div class="score">'
            f'{result1["runs"]}/{result1["wickets"]}'
            f'</div>',
            unsafe_allow_html=True
        )
        st.write(f"Overs: {result1['overs']}")

    with col2:
        st.subheader(f"🟠 {name2}")
        st.markdown(
            f'<div class="score">'
            f'{result2["runs"]}/{result2["wickets"]}'
            f'</div>',
            unsafe_allow_html=True
        )
        st.write(f"Overs: {result2['overs']}")

    # ============================================================
    # WINNER
    # ============================================================
    # The engine returns win_type/result, so a chase is displayed as
    # "wins by X wickets", while a first-innings win is "wins by X runs".
    if match_result:
        winner = match_result.get("winner")
        result_text = match_result.get("result", "")
        win_type = match_result.get("win_type")

        if winner == "TIE" or win_type == "tie":
            st.markdown(
                '<div class="winner">'
                '🤝 MATCH TIED'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            icon = "🔵" if winner == "TEAM 1" else "🟠"

            if win_type == "wickets":
                margin_text = f"by {match_result.get('margin', 0)} wickets"
            else:
                margin_text = f"by {match_result.get('margin', 0)} runs"

            st.markdown(
                f'<div class="winner">'
                f'🏆 {icon} {winner} WINS'
                f'<br>'
                f'<small>{margin_text}</small>'
                f'</div>',
                unsafe_allow_html=True
            )
    else:
        # Safe fallback.
        if result1["runs"] > result2["runs"]:
            margin = result1["runs"] - result2["runs"]
            st.markdown(
                f'<div class="winner">'
                f'🏆 🔵 {name1} WINS'
                f'<br><small>by {margin} runs</small>'
                f'</div>',
                unsafe_allow_html=True
            )
        elif result2["runs"] > result1["runs"]:
            margin = result2["runs"] - result1["runs"]
            st.markdown(
                f'<div class="winner">'
                f'🏆 🟠 {name2} WINS'
                f'<br><small>by {margin} runs</small>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="winner">🤝 MATCH TIED</div>',
                unsafe_allow_html=True
            )

    # ============================================================
    # SCORECARDS
    # ============================================================

    st.divider()

    st.header(f"🔵 {name1} Batting")
    st.dataframe(
        result1["scorecard"],
        use_container_width=True,
        hide_index=True
    )


# ================================================================
# AUCTION PAGE
# ================================================================

def auction_page():

    if st.button(
        "⬅️ Back Home"
    ):

        reset_auction()

        st.session_state.page = "home"

        st.rerun()


    st.markdown(
        '<div class="main-title">'
        '🔨 VERSUS Player Auction'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="subtitle">'
        'Build two 15-player squads through a live 30-player auction.'
        '</div>',
        unsafe_allow_html=True
    )


    # ============================================================
    # STEP 1 — SELECT 30 PLAYERS

    if not st.session_state.auction_started:

        st.header(
            "👥 Step 1 — Select 30 Players"
        )

        st.write(
            "Choose exactly 30 different players manually, or let VERSUS "
            "randomly build a popular, role-balanced 30-player auction pool."
        )

        # ------------------------------------------------------------
        # OPTION 1 — MANUAL SELECTION (UNCHANGED)
        # ------------------------------------------------------------
        selected_30 = st.multiselect(
            "Choose 30 players",
            all_players,
            max_selections=30,
            key="auction_selection"
        )

        st.info(
            f"Selected manually: {len(selected_30)}/30"
        )

        manual_col, random_col = st.columns(2)

        with manual_col:
            if st.button(
                "🎡 START AUCTION WITH SELECTED 30",
                type="primary",
                use_container_width=True,
                key="start_manual_auction"
            ):
                if len(selected_30) != 30:
                    st.error(
                        "You must select exactly 30 players."
                    )
                    return

                reset_auction()
                start_auction(selected_30)
                st.rerun()

        # ------------------------------------------------------------
        # OPTION 2 — RANDOM POPULAR + ROLE-BALANCED 30
        # ------------------------------------------------------------
        with random_col:
            if st.button(
                "🎲 PICK 30 TOP-50 PLAYERS RANDOMLY",
                use_container_width=True,
                key="random_balanced_30"
            ):
                try:
                    random_30 = get_random_balanced_auction_players(30)

                    # Store the chosen pool so the user can see exactly
                    # which 30 entered this auction.
                    st.session_state.random_auction_pool = list(random_30)

                    reset_auction()
                    st.session_state.random_auction_pool = list(random_30)
                    start_auction(random_30)
                    st.rerun()

                except Exception:
                    st.error("❌ Could not create the random 30-player pool.")
                    st.code(traceback.format_exc())

        st.caption(
            "Random mode ranks players by IPL matches played, takes the Top 50 in "
            "each role, then randomly selects exactly 12 batsmen/WKs, 6 "
            "all-rounders and 12 bowlers. The 30 are shuffled before auction."
        )

        return


    # ============================================================
    # AUCTION HEADER
    # ============================================================

    team1_count = len(
        st.session_state.team1_squad
    )

    team2_count = len(
        st.session_state.team2_squad
    )


    col1, col2 = st.columns(2)


    with col1:

        st.subheader(
            "🔵 TEAM 1"
        )

        st.metric(
            "Players",
            f"{team1_count}/15"
        )

        st.metric(
            "Remaining Purse",
            f"₹{st.session_state.team1_purse:.2f} Cr"
        )


    with col2:

        st.subheader(
            "🟠 TEAM 2"
        )

        st.metric(
            "Players",
            f"{team2_count}/15"
        )

        st.metric(
            "Remaining Purse",
            f"₹{st.session_state.team2_purse:.2f} Cr"
        )


    st.divider()


    # ============================================================
    # FINISHED
    # ============================================================

    if st.session_state.auction_finished:

        show_finished_auction()

        return


    # ============================================================
    # AUTO FINISH / ASSIGN
    # ============================================================

    auto_finish_if_needed()


    if st.session_state.auction_finished:

        show_finished_auction()

        return


    # ============================================================
    # SPIN WHEEL
    # ============================================================

    if st.session_state.current_player is None:

        st.header(
            "🎡 Player Wheel"
        )


        remaining = (
            len(
                st.session_state.auction_players
            )
            -
            st.session_state.auction_index
        )


        st.write(
            f"Players remaining in wheel: **{remaining}**"
        )


        if st.button(
            "🎡 SPIN WHEEL",
            type="primary",
            use_container_width=True
        ):

            spin_wheel()

            st.rerun()


        # --------------------------------------------------------
        # Show randomized order privately as progress
        # --------------------------------------------------------

        st.caption(
            "The next player is selected randomly from "
            "the remaining 30-player pool."
        )


        return


    # ============================================================
    # CURRENT PLAYER
    # ============================================================

    player = (
        st.session_state.current_player
    )


    auction_position = (
        st.session_state.auction_index
    )


    st.header(
        f"🎡 Player {auction_position} / 30"
    )


    st.markdown(
        f'<div class="auction-player">'
        f'🏏 {player}'
        f'</div>',
        unsafe_allow_html=True
    )


    # ============================================================
    # CURRENT BID
    # ============================================================

    st.markdown(
        f'<div class="bid-price">'
        f'Current Bid: '
        f'₹{st.session_state.current_bid:.2f} Cr'
        f'</div>',
        unsafe_allow_html=True
    )


    if st.session_state.highest_bidder:

        st.info(
            f"🔨 Current highest bidder: "
            f"{st.session_state.highest_bidder}"
        )

    else:

        st.info(
            "No bid yet. First bid starts at ₹1 Cr."
        )


    if st.session_state.auction_message:

        st.success(
            st.session_state.auction_message
        )


    st.divider()


    # ============================================================
    # BIDDING SIDE BY SIDE
    # ============================================================

    st.header(
        "💰 Bidding"
    )


    col1, col2 = st.columns(2)


    # ============================================================
    # TEAM 1
    # ============================================================

    with col1:

        st.subheader(
            "🔵 TEAM 1"
        )


        team1_full = (
            len(
                st.session_state.team1_squad
            )
            >=
            15
        )


        if team1_full:

            st.warning(
                "🔒 Team 1 already has 15 players."
            )


        elif st.session_state.team1_stopped:

            st.warning(
                "🛑 Team 1 stopped bidding on this player."
            )


        else:

            b1, b2 = st.columns(2)

            with b1:

                if st.button(
                    "+₹1 Cr",
                    key="team1_1",
                    use_container_width=True
                ):

                    place_bid(
                        "TEAM 1",
                        1
                    )

                    st.rerun()


            with b2:

                if st.button(
                    "+₹2 Cr",
                    key="team1_2",
                    use_container_width=True
                ):

                    place_bid(
                        "TEAM 1",
                        2
                    )

                    st.rerun()


            b3, b4 = st.columns(2)

            with b3:

                if st.button(
                    "+₹5 Cr",
                    key="team1_5",
                    use_container_width=True
                ):

                    place_bid(
                        "TEAM 1",
                        5
                    )

                    st.rerun()


            with b4:

                if st.button(
                    "+₹10 Cr",
                    key="team1_10",
                    use_container_width=True
                ):

                    place_bid(
                        "TEAM 1",
                        10
                    )

                    st.rerun()


            if st.button(
                "🛑 STOP BIDDING",
                key="team1_stop",
                use_container_width=True
            ):

                stop_bidding(
                    "TEAM 1"
                )

                st.rerun()


    # ============================================================
    # TEAM 2
    # ============================================================

    with col2:

        st.subheader(
            "🟠 TEAM 2"
        )


        team2_full = (
            len(
                st.session_state.team2_squad
            )
            >=
            15
        )


        if team2_full:

            st.warning(
                "🔒 Team 2 already has 15 players."
            )


        elif st.session_state.team2_stopped:

            st.warning(
                "🛑 Team 2 stopped bidding on this player."
            )


        else:

            b1, b2 = st.columns(2)

            with b1:

                if st.button(
                    "+₹1 Cr",
                    key="team2_1",
                    use_container_width=True
                ):

                    place_bid(
                        "TEAM 2",
                        1
                    )

                    st.rerun()


            with b2:

                if st.button(
                    "+₹2 Cr",
                    key="team2_2",
                    use_container_width=True
                ):

                    place_bid(
                        "TEAM 2",
                        2
                    )

                    st.rerun()


            b3, b4 = st.columns(2)

            with b3:

                if st.button(
                    "+₹5 Cr",
                    key="team2_5",
                    use_container_width=True
                ):

                    place_bid(
                        "TEAM 2",
                        5
                    )

                    st.rerun()


            with b4:

                if st.button(
                    "+₹10 Cr",
                    key="team2_10",
                    use_container_width=True
                ):

                    place_bid(
                        "TEAM 2",
                        10
                    )

                    st.rerun()


            if st.button(
                "🛑 STOP BIDDING",
                key="team2_stop",
                use_container_width=True
            ):

                stop_bidding(
                    "TEAM 2"
                )

                st.rerun()


    # ============================================================
    # PLAYER DECISION
    # ============================================================

    st.divider()


    st.header(
        "🏷️ Player Decision"
    )


    highest = (
        st.session_state.highest_bidder
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        disabled = (

            highest != "TEAM 1"

            or

            len(
                st.session_state.team1_squad
            ) >= 11
        )


        if st.button(
            "🔵 SOLD TO TEAM 1",
            use_container_width=True,
            disabled=disabled
        ):

            complete_player(
                "TEAM 1"
            )

            st.rerun()


    with col2:

        disabled = (

            highest != "TEAM 2"

            or

            len(
                st.session_state.team2_squad
            ) >= 11
        )


        if st.button(
            "🟠 SOLD TO TEAM 2",
            use_container_width=True,
            disabled=disabled
        ):

            complete_player(
                "TEAM 2"
            )

            st.rerun()


    with col3:

        if st.button(
            "⏭️ PASS PLAYER",
            use_container_width=True
        ):

            pass_player()

            st.rerun()


    # ============================================================
    # CURRENT SQUADS
    # ============================================================

    st.divider()


    st.header(
        "📋 Current Squads"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.subheader(
            f"🔵 Team 1 — {team1_count}/11"
        )

        if st.session_state.team1_squad:

            for i, player in enumerate(
                st.session_state.team1_squad,
                1
            ):

                st.write(
                    f"{i}. {player}"
                )

        else:

            st.caption(
                "No players purchased yet."
            )


    with col2:

        st.subheader(
            f"🟠 Team 2 — {team2_count}/11"
        )

        if st.session_state.team2_squad:

            for i, player in enumerate(
                st.session_state.team2_squad,
                1
            ):

                st.write(
                    f"{i}. {player}"
                )

        else:

            st.caption(
                "No players purchased yet."
            )


# ================================================================
# FINISHED AUCTION
# ================================================================

def show_finished_auction():
    team1 = st.session_state.team1_squad
    team2 = st.session_state.team2_squad

    if len(team1) < 11 or len(team2) < 11:
        st.error(
            f"Auction cannot finish yet. "
            f"Team 1: {len(team1)}/15 | "
            f"Team 2: {len(team2)}/15"
        )
        return

    st.success("🎉 AUCTION COMPLETE!")

    st.header("🏏 Final Auction Squads")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔵 TEAM 1 — 15/15")
        st.metric(
            "Remaining Purse",
            f"₹{st.session_state.team1_purse:.2f} Cr"
        )

        for i, player in enumerate(team1, 1):
            st.write(f"**{i}.** {player}")

    with col2:
        st.subheader("🟠 TEAM 2 — 15/15")
        st.metric(
            "Remaining Purse",
            f"₹{st.session_state.team2_purse:.2f} Cr"
        )

        for i, player in enumerate(team2, 1):
            st.write(f"**{i}.** {player}")

    # ============================================================
    # AUCTION HISTORY
    # ============================================================

    st.divider()
    st.header("📜 Complete Auction")

    if st.session_state.auction_history:
        auction_df = pd.DataFrame(
            st.session_state.auction_history
        )
        st.dataframe(
            auction_df,
            use_container_width=True,
            hide_index=True
        )

    # ============================================================
    # PLAYING XI + BATTING ORDER
    # ============================================================

    st.divider()
    st.header("🏏 Select Playing XI & Batting Order")

    st.info(
        "Your auction squad contains 15 players. "
        "Click the players below in the exact order you want them to bat. "
        "The first player you click becomes #1, the second becomes #2, and so on."
    )

    # ------------------------------------------------------------
    # IMPORTANT:
    # The auction gives each team exactly 15 players.
    # We do NOT use another multiselect here.
    #
    # The 15 players are displayed as individual buttons.
    # Clicking a player adds that player to the Playing XI in
    # click order. This click order becomes the batting order.
    # ------------------------------------------------------------

    if "auction_playing11_team1" not in st.session_state:
        st.session_state.auction_playing11_team1 = []

    if "auction_playing11_team2" not in st.session_state:
        st.session_state.auction_playing11_team2 = []

    def toggle_playing_player(team_key, player):
        selected = st.session_state[team_key]

        # If already selected, remove the player.
        # The remaining players keep their existing order.
        if player in selected:
            st.session_state[team_key] = [
                p for p in selected if p != player
            ]
            return

        # Otherwise add to the end = next batting position.
        if len(selected) < 11:
            st.session_state[team_key] = selected + [player]

    col1, col2 = st.columns(2)

    # ============================================================
    # TEAM 1
    # ============================================================

    with col1:
        st.subheader("🔵 Team 1 — Squad of 15")

        selected1 = st.session_state.auction_playing11_team1

        st.caption(
            f"Click in batting order • Playing XI: {len(selected1)}/11"
        )

        if len(selected1) == 11:
            st.success("✅ Team 1 Playing XI complete.")

        # Show the 15 auctioned players as buttons.
        # No dropdown is used.
        for row_start in range(0, len(team1), 3):
            row_players = team1[row_start:row_start + 3]
            cols = st.columns(3)

            for j, player in enumerate(row_players):
                with cols[j]:
                    if player in selected1:
                        position = selected1.index(player) + 1
                        st.button(
                            f"🟢 {position}. {player}",
                            key=f"auction_t1_player_{row_start+j}",
                            use_container_width=True,
                            on_click=toggle_playing_player,
                            args=("auction_playing11_team1", player),
                        )
                    else:
                        st.button(
                            player,
                            key=f"auction_t1_player_{row_start+j}",
                            use_container_width=True,
                            disabled=(len(selected1) >= 11),
                            on_click=toggle_playing_player,
                            args=("auction_playing11_team1", player),
                        )

        st.markdown("**Team 1 Batting Order**")

        if selected1:
            for pos, player in enumerate(selected1, 1):
                st.write(f"**{pos}.** {player}")
        else:
            st.caption("No players selected yet.")

        if selected1:
            if st.button(
                "↩️ Clear Team 1 Playing XI",
                key="clear_auction_t1_xi",
                use_container_width=True,
            ):
                st.session_state.auction_playing11_team1 = []
                st.rerun()

    # ============================================================
    # TEAM 2
    # ============================================================

    with col2:
        st.subheader("🟠 Team 2 — Squad of 15")

        selected2 = st.session_state.auction_playing11_team2

        st.caption(
            f"Click in batting order • Playing XI: {len(selected2)}/11"
        )

        if len(selected2) == 11:
            st.success("✅ Team 2 Playing XI complete.")

        # Show the 15 auctioned players as buttons.
        # No dropdown is used.
        for row_start in range(0, len(team2), 3):
            row_players = team2[row_start:row_start + 3]
            cols = st.columns(3)

            for j, player in enumerate(row_players):
                with cols[j]:
                    if player in selected2:
                        position = selected2.index(player) + 1
                        st.button(
                            f"🟠 {position}. {player}",
                            key=f"auction_t2_player_{row_start+j}",
                            use_container_width=True,
                            on_click=toggle_playing_player,
                            args=("auction_playing11_team2", player),
                        )
                    else:
                        st.button(
                            player,
                            key=f"auction_t2_player_{row_start+j}",
                            use_container_width=True,
                            disabled=(len(selected2) >= 11),
                            on_click=toggle_playing_player,
                            args=("auction_playing11_team2", player),
                        )

        st.markdown("**Team 2 Batting Order**")

        if selected2:
            for pos, player in enumerate(selected2, 1):
                st.write(f"**{pos}.** {player}")
        else:
            st.caption("No players selected yet.")

        if selected2:
            if st.button(
                "↩️ Clear Team 2 Playing XI",
                key="clear_auction_t2_xi",
                use_container_width=True,
            ):
                st.session_state.auction_playing11_team2 = []
                st.rerun()

    # ============================================================
    # FINAL BATTING ORDER SUMMARY
    # ============================================================

    if (
        len(st.session_state.auction_playing11_team1) == 11
        and
        len(st.session_state.auction_playing11_team2) == 11
    ):
        st.divider()
        st.subheader("📋 Final Playing XI & Batting Orders")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🔵 Team 1")
            for i, player in enumerate(
                st.session_state.auction_playing11_team1, 1
            ):
                st.write(f"**{i}.** {player}")

        with col2:
            st.markdown("### 🟠 Team 2")
            for i, player in enumerate(
                st.session_state.auction_playing11_team2, 1
            ):
                st.write(f"**{i}.** {player}")

    # ============================================================
    # BOWLING ATTACK
    # ============================================================

    st.divider()
    st.header("🎯 Select Bowling Attacks")

    st.caption(
        "Choose exactly 5 bowling options from each final playing XI."
    )

    team1_playing = st.session_state.auction_playing11_team1
    team2_playing = st.session_state.auction_playing11_team2

    team1_possible = safe_find_bowlers(team1_playing)
    team2_possible = safe_find_bowlers(team2_playing)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔵 Team 1 Bowling")

        if team1_possible:
            auction_team1_bowlers = st.multiselect(
                "Team 1 Bowlers",
                team1_possible,
                max_selections=5,
                key="auction_bowlers_1"
            )
        else:
            auction_team1_bowlers = []
            st.warning("No bowling options detected.")

    with col2:
        st.subheader("🟠 Team 2 Bowling")

        if team2_possible:
            auction_team2_bowlers = st.multiselect(
                "Team 2 Bowlers",
                team2_possible,
                max_selections=5,
                key="auction_bowlers_2"
            )
        else:
            auction_team2_bowlers = []
            st.warning("No bowling options detected.")

    # ============================================================
    # TOSS — AFTER PLAYING XI / BOWLING SELECTION
    # ============================================================

    auction_toss_signature = (
        tuple(st.session_state.auction_playing11_team1),
        tuple(st.session_state.auction_playing11_team2),
        tuple(auction_team1_bowlers),
        tuple(auction_team2_bowlers)
    )

    toss_winner, toss_decision = toss_section(
        "auction",
        "TEAM 1",
        "TEAM 2",
        auction_toss_signature
    )

    # ============================================================
    # SETTINGS
    # ============================================================

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        auction_overs = st.number_input(
            "Match Overs",
            min_value=1,
            max_value=20,
            value=20,
            step=1,
            key="auction_match_overs"
        )

    with col2:
        st.info(
            "🎲 Every prediction uses fresh random simulation. "
            "The same teams can produce a different result each time."
        )

    # ============================================================
    # PREDICT
    # ============================================================

    st.divider()
    st.header("🚀 Ready for ML Prediction?")

    st.write(
        "The ML simulation will use the exact 11-player Playing XI and batting order shown above."
    )

    if st.button(
        "🏏 PREDICT MATCH",
        type="primary",
        use_container_width=True
    ):

        if len(st.session_state.auction_playing11_team1) != 11:
            st.error("Team 1 must have exactly 11 players in the Playing XI.")
            return

        if len(st.session_state.auction_playing11_team2) != 11:
            st.error("Team 2 must have exactly 11 players in the Playing XI.")
            return

        if len(auction_team1_bowlers) < 5:
            st.error("Select at least 5 Team 1 bowlers.")
            return

        if len(auction_team2_bowlers) < 5:
            st.error("Select at least 5 Team 2 bowlers.")
            return

        if toss_winner is None or toss_decision is None:
            st.error(
                "Complete the toss and let the toss winner choose BAT or BOWL before predicting."
            )
            return

        # IMPORTANT:
        # Do NOT sort these lists and do NOT replace them with the auction
        # purchase order. The user's selected batting order is passed directly.
        batting_team1 = list(st.session_state.auction_playing11_team1)
        batting_team2 = list(st.session_state.auction_playing11_team2)

        # IMPORTANT: never reuse a fixed seed here.
        # Every press of PREDICT MATCH starts a fresh stochastic simulation.
        np.random.seed(None)

        with st.spinner(
            "🤖 Running ML ball-by-ball simulation using your batting order..."
        ):
            try:
                match_result = simulate_match(
                    batting_team1,
                    auction_team1_bowlers,
                    batting_team2,
                    auction_team2_bowlers,
                    int(auction_overs),
                    toss_winner=toss_winner,
                    toss_decision=toss_decision
                )

                result1 = match_result["team1"]
                result2 = match_result["team2"]

            except Exception:
                st.error("❌ ML prediction failed.")
                st.code(traceback.format_exc())
                return

        st.session_state.prediction = True

        display_match_result(
            result1,
            result2,
            "TEAM 1",
            "TEAM 2",
            match_result
        )

    # ============================================================
    # RESET AUCTION
    # ============================================================

    st.divider()

    if st.button(
        "🔄 Start New Auction",
        use_container_width=True
    ):
        reset_auction()
        st.rerun()

# ================================================================
# ROUTER
# ================================================================

if st.session_state.page == "home":

    home_page()


elif st.session_state.page == "custom":

    custom_page()


elif st.session_state.page == "auction":

    auction_page()


else:

    st.session_state.page = "home"

    st.rerun()


# ================================================================
# FOOTER
# ================================================================

st.markdown("---")

st.caption(
    "VERSUS • IPL Custom XI & Auction Simulator • "
    "Historical batter-vs-bowler data + ML stochastic simulation"
)