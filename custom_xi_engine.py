import os
import joblib
import numpy as np
import pandas as pd


# ================================================================
# VERSUS CUSTOM XI ENGINE
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

    raise FileNotFoundError(
        f"{filename} not found inside {BASE_DIR}"
    )


# ================================================================
# LOAD MODELS
# ================================================================

runs_model = joblib.load(
    find_file("custom_xi_runs_model.pkl")
)

balls_model = joblib.load(
    find_file("custom_xi_balls_model.pkl")
)

dismissal_model = joblib.load(
    find_file("custom_xi_dismissal_model.pkl")
)

MODEL_FEATURES = joblib.load(
    find_file("custom_xi_model_features.pkl")
)

matchup_df = pd.read_pickle(
    find_file("matchup_df.pkl")
)


# ================================================================
# NORMALIZE DATA
# ================================================================

matchup_df["batter"] = (
    matchup_df["batter"]
    .astype(str)
    .str.strip()
)

matchup_df["bowler"] = (
    matchup_df["bowler"]
    .astype(str)
    .str.strip()
)


# ================================================================
# FIND BOWLERS
# ================================================================
#
# IMPORTANT:
# app.py requires this function.
# ================================================================

def find_bowlers(players):

    players = list(
        dict.fromkeys(players)
    )

    if len(players) == 0:
        return []

    result = []

    # ------------------------------------------------------------
    # Try player_profile.pkl
    # ------------------------------------------------------------

    try:

        profile_path = find_file(
            "player_profile.pkl"
        )

        profile = pd.read_pickle(
            profile_path
        )

        # --------------------------------------------------------
        # Find player-name column
        # --------------------------------------------------------

        name_column = None

        for col in [
            "player",
            "Player",
            "name",
            "Name"
        ]:

            if col in profile.columns:

                name_column = col
                break


        if name_column is not None:

            for player in players:

                rows = profile[
                    profile[name_column]
                    .astype(str)
                    .str.strip()
                    ==
                    str(player).strip()
                ]

                if len(rows) == 0:
                    continue

                row = rows.iloc[0]

                role_text = ""

                for col in [
                    "role",
                    "Role",
                    "playing_role",
                    "Playing Role",
                    "type",
                    "Type"
                ]:

                    if col in profile.columns:

                        role_text += " " + str(
                            row.get(col, "")
                        ).lower()

                if (
                    "bowl" in role_text
                    or
                    "all-round" in role_text
                    or
                    "all round" in role_text
                    or
                    "allround" in role_text
                ):

                    result.append(player)


    except Exception:

        pass


    # ------------------------------------------------------------
    # Historical bowling fallback
    # ------------------------------------------------------------

    for player in players:

        if player in result:
            continue

        try:

            rows = matchup_df[
                matchup_df["bowler"]
                ==
                str(player).strip()
            ]

            if len(rows) > 0:

                result.append(player)

        except Exception:

            pass


    # ------------------------------------------------------------
    # Return unique players
    # ------------------------------------------------------------

    return list(
        dict.fromkeys(result)
    )


# ================================================================
# HISTORY
# ================================================================

def get_history(
    batter,
    bowler
):

    rows = matchup_df[
        (
            matchup_df["batter"]
            ==
            str(batter).strip()
        )
        &
        (
            matchup_df["bowler"]
            ==
            str(bowler).strip()
        )
    ]

    if len(rows) == 0:

        rows = matchup_df[
            matchup_df["batter"]
            ==
            str(batter).strip()
        ]

    if len(rows) == 0:

        return {
            "previous_balls": 0,
            "previous_runs": 0,
            "previous_dismissals": 0,
            "previous_fours": 0,
            "previous_sixes": 0,
            "previous_strike_rate": 0,
            "last_5_runs": 0,
            "last_5_balls": 0,
            "last_5_strike_rate": 0,
            "last_5_dismissals": 0
        }

    row = rows.iloc[-1]

    def num(key):

        try:

            value = float(
                row.get(key, 0)
            )

            if np.isfinite(value):
                return value

        except Exception:

            pass

        return 0.0


    return {

        "previous_balls":
            num("previous_balls"),

        "previous_runs":
            num("previous_runs"),

        "previous_dismissals":
            num("previous_dismissals"),

        "previous_fours":
            num("previous_fours"),

        "previous_sixes":
            num("previous_sixes"),

        "previous_strike_rate":
            num("previous_strike_rate"),

        "last_5_runs":
            num("last_5_runs"),

        "last_5_balls":
            num("last_5_balls"),

        "last_5_strike_rate":
            num("last_5_strike_rate"),

        "last_5_dismissals":
            num("last_5_dismissals")
    }


# ================================================================
# FEATURES
# ================================================================

def make_features(
    batter,
    bowler
):

    h = get_history(
        batter,
        bowler
    )

    previous_balls = h[
        "previous_balls"
    ]

    previous_runs = h[
        "previous_runs"
    ]

    previous_dismissals = h[
        "previous_dismissals"
    ]

    previous_fours = h[
        "previous_fours"
    ]

    previous_sixes = h[
        "previous_sixes"
    ]

    previous_sr = h[
        "previous_strike_rate"
    ]

    last_5_runs = h[
        "last_5_runs"
    ]

    last_5_balls = h[
        "last_5_balls"
    ]

    last_5_sr = h[
        "last_5_strike_rate"
    ]

    last_5_dismissals = h[
        "last_5_dismissals"
    ]


    previous_run_rate = (

        previous_runs
        /
        previous_balls

        if previous_balls > 0

        else 0
    )


    last_5_run_rate = (

        last_5_runs
        /
        last_5_balls

        if last_5_balls > 0

        else 0
    )


    previous_boundary_rate = (

        (
            previous_fours
            +
            previous_sixes
        )
        /
        previous_balls

        if previous_balls > 0

        else 0
    )


    recent_boundary_rate = (

        last_5_runs
        /
        last_5_balls

        if last_5_balls > 0

        else 0
    )


    dismissal_rate_history = (

        previous_dismissals
        /
        previous_balls

        if previous_balls > 0

        else 0
    )


    values = {

        "previous_balls":
            previous_balls,

        "previous_runs":
            previous_runs,

        "previous_dismissals":
            previous_dismissals,

        "previous_fours":
            previous_fours,

        "previous_sixes":
            previous_sixes,

        "previous_strike_rate":
            previous_sr,

        "last_5_runs":
            last_5_runs,

        "last_5_balls":
            last_5_balls,

        "last_5_strike_rate":
            last_5_sr,

        "last_5_dismissals":
            last_5_dismissals,

        "previous_run_rate":
            previous_run_rate,

        "last_5_run_rate":
            last_5_run_rate,

        "previous_boundary_rate":
            previous_boundary_rate,

        "recent_boundary_rate":
            recent_boundary_rate,

        "dismissal_rate_history":
            dismissal_rate_history
    }


    X = pd.DataFrame(
        [values]
    )


    for column in MODEL_FEATURES:

        if column not in X.columns:

            X[column] = 0


    X = X[
        MODEL_FEATURES
    ]

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(0)

    return X


# ================================================================
# WICKET PROBABILITY
# ================================================================

def wicket_probability(
    batter,
    bowler,
    X
):

    probability = None


    if hasattr(
        dismissal_model,
        "predict_proba"
    ):

        try:

            p = dismissal_model.predict_proba(X)

            if (
                len(p.shape) == 2
                and
                p.shape[1] >= 2
            ):

                probability = float(
                    p[0][1]
                )

        except Exception:

            probability = None


    if probability is None:

        try:

            probability = float(
                dismissal_model.predict(X)[0]
            )

        except Exception:

            probability = 0.025


    return float(
        np.clip(
            probability,
            0.005,
            0.075
        )
    )


# ================================================================
# MATCHUP CACHE
# ================================================================

def build_cache(
    batting_xi,
    bowling_xi
):

    cache = {}


    for batter in batting_xi:

        for bowler in bowling_xi:

            X = make_features(
                batter,
                bowler
            )


            try:

                runs_per_ball = float(
                    runs_model.predict(X)[0]
                )

            except Exception:

                runs_per_ball = 1.0


            runs_per_ball = float(
                np.clip(
                    runs_per_ball,
                    0.1,
                    3.5
                )
            )


            cache[
                (batter, bowler)
            ] = {

                "rpb":
                    runs_per_ball,

                "wicket":
                    wicket_probability(
                        batter,
                        bowler,
                        X
                    )
            }


    return cache


# ================================================================
# CHOOSE BOWLER
# ================================================================

def choose_bowler(
    striker,
    bowling_xi,
    cache,
    over_counts
):

    available = [

        bowler

        for bowler in bowling_xi

        if over_counts.get(
            bowler,
            0
        ) < 4
    ]


    if len(available) == 0:

        available = list(
            bowling_xi
        )


    weights = []


    for bowler in available:

        info = cache[
            (striker, bowler)
        ]


        weight = (

            1.0
            +
            info["wicket"] * 4
            +
            max(
                0,
                1.5 - info["rpb"]
            ) * 0.2
        )


        weights.append(
            weight
        )


    weights = np.array(
        weights,
        dtype=float
    )


    weights /= weights.sum()


    return np.random.choice(
        available,
        p=weights
    )


# ================================================================
# ONE BALL
# ================================================================

def predict_ball(
    batter,
    bowler,
    cache
):

    info = cache[
        (batter, bowler)
    ]


    rpb = info[
        "rpb"
    ]


    wicket_rate = info[
        "wicket"
    ]


    possible = np.array(
        [0, 1, 2, 3, 4, 6]
    )


    distance = np.abs(
        possible - rpb
    )


    weights = np.exp(
        -distance / 0.75
    )


    weights /= weights.sum()


    runs = int(
        np.random.choice(
            possible,
            p=weights
        )
    )


    wicket = (
        np.random.random()
        <
        wicket_rate
    )


    return runs, wicket


# ================================================================
# SIMULATE INNINGS
# ================================================================

def simulate_innings(
    batting_xi,
    bowling_xi,
    overs=20,
    target=None
):

    batting_xi = list(
        dict.fromkeys(
            batting_xi
        )
    )

    bowling_xi = list(
        dict.fromkeys(
            bowling_xi
        )
    )


    if len(batting_xi) != 11:

        raise ValueError(
            "Batting XI must contain exactly 11 players."
        )


    if len(bowling_xi) != 5:

        raise ValueError(
            "Exactly 5 bowling options are required."
        )


    overs = int(
        overs
    )


    cache = build_cache(
        batting_xi,
        bowling_xi
    )


    batting = {

        player: {

            "runs": 0,
            "balls": 0,
            "fours": 0,
            "sixes": 0,
            "status": "DID NOT BAT",
            "dismissed_by": ""

        }

        for player in batting_xi
    }


    bowling = {

        player: {

            "balls": 0,
            "runs": 0,
            "wickets": 0

        }

        for player in bowling_xi
    }


    over_counts = {

        player: 0

        for player in bowling_xi
    }


    striker = batting_xi[0]

    non_striker = batting_xi[1]

    next_batter = 2


    batting[striker][
        "status"
    ] = "NOT OUT"


    batting[non_striker][
        "status"
    ] = "NOT OUT"


    total_runs = 0

    wickets = 0

    legal_balls = 0

    maximum_balls = overs * 6


    # ============================================================
    # OVERS
    # ============================================================

    while (
        legal_balls < maximum_balls
        and wickets < 10
        and (target is None or total_runs < target)
    ):

        # --------------------------------------------------------
        # SELECT BOWLER ONCE
        # --------------------------------------------------------

        bowler = choose_bowler(
            striker,
            bowling_xi,
            cache,
            over_counts
        )


        balls_this_over = 0


        # --------------------------------------------------------
        # FULL 6 BALL OVER
        # --------------------------------------------------------

        while (

            balls_this_over < 6

            and

            legal_balls < maximum_balls

            and

            wickets < 10
        ):

            runs, wicket = predict_ball(
                striker,
                bowler,
                cache
            )


            # ----------------------------------------------------
            # BATTER
            # ----------------------------------------------------

            batting[striker][
                "runs"
            ] += runs

            batting[striker][
                "balls"
            ] += 1


            if runs == 4:

                batting[striker][
                    "fours"
                ] += 1


            if runs == 6:

                batting[striker][
                    "sixes"
                ] += 1


            # ----------------------------------------------------
            # TEAM
            # ----------------------------------------------------

            total_runs += runs

            legal_balls += 1

            balls_this_over += 1


            # ----------------------------------------------------
            # BOWLER
            # ----------------------------------------------------

            bowling[bowler][
                "balls"
            ] += 1

            bowling[bowler][
                "runs"
            ] += runs

            # ----------------------------------------------------
            # CHASE TARGET
            # ----------------------------------------------------
            # If this is the second innings and the target has
            # been reached, end the innings immediately.
            if target is not None and total_runs >= target:
                break


            # ----------------------------------------------------
            # WICKET
            # ----------------------------------------------------

            if wicket:

                wickets += 1


                batting[striker][
                    "status"
                ] = "OUT"


                batting[striker][
                    "dismissed_by"
                ] = bowler


                bowling[bowler][
                    "wickets"
                ] += 1


                if next_batter < 11:

                    striker = batting_xi[
                        next_batter
                    ]

                    next_batter += 1


                    batting[striker][
                        "status"
                    ] = "NOT OUT"

                else:

                    break


            else:

                if runs % 2 == 1:

                    striker, non_striker = (

                        non_striker,
                        striker
                    )


        # --------------------------------------------------------
        # COMPLETE OVER
        # --------------------------------------------------------

        over_counts[
            bowler
        ] += 1


        if (

            balls_this_over == 6

            and

            wickets < 10

            and

            legal_balls < maximum_balls
        ):

            striker, non_striker = (

                non_striker,
                striker
            )


    # ============================================================
    # FINAL BATTER STATUS
    # ============================================================

    for player in batting_xi:

        if batting[player]["balls"] == 0:

            batting[player][
                "status"
            ] = "DID NOT BAT"

        elif batting[player][
            "status"
        ] != "OUT":

            batting[player][
                "status"
            ] = "NOT OUT"


    # ============================================================
    # BATTING CARD
    # ============================================================

    batting_card = []


    for number, player in enumerate(
        batting_xi,
        1
    ):

        data = batting[player]


        sr = (

            data["runs"]
            /
            data["balls"]
            *
            100

            if data["balls"] > 0

            else 0
        )


        batting_card.append({

            "#":
                number,

            "Batter":
                player,

            "Runs":
                data["runs"],

            "Balls":
                data["balls"],

            "4s":
                data["fours"],

            "6s":
                data["sixes"],

            "SR":
                round(sr, 1),

            "Status":
                data["status"],

            "Dismissed By":
                data["dismissed_by"]
        })


    # ============================================================
    # BOWLING CARD
    # ============================================================

    bowling_card = []


    for player in bowling_xi:

        data = bowling[player]


        if data["balls"] == 0:
            continue


        bowling_card.append({

            "Bowler":
                player,

            "Overs":
                f"{data['balls'] // 6}."
                f"{data['balls'] % 6}",

            "Runs":
                data["runs"],

            "Wickets":
                data["wickets"]
        })


    return {

        "runs":
            int(total_runs),

        "wickets":
            int(wickets),

        "balls":
            int(legal_balls),

        "overs":
            f"{legal_balls // 6}."
            f"{legal_balls % 6}",

        "scorecard":
            pd.DataFrame(
                batting_card
            ),

        "bowling":
            pd.DataFrame(
                bowling_card
            )
    }


# ================================================================
# SIMULATE MATCH
# ================================================================

def simulate_match(
    team1_batting,
    team1_bowling,
    team2_batting,
    team2_bowling,
    overs=20,
    toss_winner=None,
    toss_decision=None
):
    """
    Simulate a complete T20 match with a real toss and chase.

    If toss_winner / toss_decision are not supplied, both are randomized.
    The toss winner chooses BAT or BOWL, and the innings order is then
    determined from that decision.
    """

    # ---------------------------
    # TOSS
    # ---------------------------
    if toss_winner not in ("TEAM 1", "TEAM 2"):
        toss_winner = "TEAM 1" if np.random.random() < 0.5 else "TEAM 2"

    if toss_decision is None:
        # A randomized toss choice keeps the simulation stochastic.
        toss_decision = "BAT" if np.random.random() < 0.5 else "BOWL"
    else:
        toss_decision = str(toss_decision).upper()
        if toss_decision not in ("BAT", "BOWL"):
            toss_decision = "BAT"

    # Determine who bats first.
    if (toss_winner == "TEAM 1" and toss_decision == "BAT") or (
        toss_winner == "TEAM 2" and toss_decision == "BOWL"
    ):
        first_batting = "TEAM 1"
    else:
        first_batting = "TEAM 2"

    # ---------------------------
    # FIRST INNINGS
    # ---------------------------
    if first_batting == "TEAM 1":
        team1 = simulate_innings(team1_batting, team2_bowling, overs)
        target = team1["runs"] + 1
        team2 = simulate_innings(
            team2_batting, team1_bowling, overs, target=target
        )
        first_name = "TEAM 1"
        second_name = "TEAM 2"
    else:
        team2 = simulate_innings(team2_batting, team1_bowling, overs)
        target = team2["runs"] + 1
        team1 = simulate_innings(
            team1_batting, team2_bowling, overs, target=target
        )
        first_name = "TEAM 2"
        second_name = "TEAM 1"

    first = team1 if first_name == "TEAM 1" else team2
    second = team1 if second_name == "TEAM 1" else team2

    # ---------------------------
    # RESULT
    # ---------------------------
    if second["runs"] >= target:
        winner = second_name
        wickets_remaining = max(0, 10 - second["wickets"])
        result = f"{winner} wins by {wickets_remaining} wickets"
        win_type = "wickets"
        margin = wickets_remaining
    elif first["runs"] > second["runs"]:
        winner = first_name
        margin = first["runs"] - second["runs"]
        result = f"{winner} wins by {margin} runs"
        win_type = "runs"
    else:
        winner = "TIE"
        margin = 0
        result = "Match tied"
        win_type = "tie"

    return {
        "team1": team1,
        "team2": team2,
        "winner": winner,
        "result": result,
        "win_type": win_type,
        "margin": margin,
        "toss_winner": toss_winner,
        "toss_decision": toss_decision,
        "first_batting": first_name,
        "second_batting": second_name,
        "target": target
    }


# ================================================================
# IMPORT TEST
# ================================================================

print(
    "CUSTOM XI ENGINE LOADED SUCCESSFULLY"
)

print(
    "find_bowlers() available:",
    callable(find_bowlers)
)

print(
    "simulate_innings() available:",
    callable(simulate_innings)
)

print(
    "simulate_match() available:",
    callable(simulate_match)
)