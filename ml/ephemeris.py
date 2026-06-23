#!/usr/bin/env python3
"""Swiss-ephemeris (financial astrology) features.

Uses the Moshier built-in ephemeris (swe.FLG_MOSEPH) so NO external ephemeris
data files are required. Computed at 21:00 UT (~US cash close) for each date.

Feature blocks per date:
  - per body: sin(lon), cos(lon), speed, retro flag
  - moon synodic phase (Sun-Moon angle) sin/cos + illumination proxy
  - selected planet-pair angular separations + hard-aspect proximity
  - count of retrograde planets
  - crude harmonic "aspect potential" indices (Bradley-ish)
"""
import math
import swisseph as swe

FLAGS = swe.FLG_MOSEPH | swe.FLG_SPEED
FLGE = swe.FLG_MOSEPH | swe.FLG_SPEED | swe.FLG_EQUATORIAL   # RA/declination
FLGH = swe.FLG_MOSEPH | swe.FLG_SPEED | swe.FLG_HELCTR       # heliocentric

# --- Bradley Siderograph construction (Bradley 1948) ---
BR_BODIES = ["sun","mercury","venus","mars","jupiter","saturn","uranus","neptune","pluto"]
BR_LONG = [("jupiter","saturn"),("jupiter","uranus"),("jupiter","neptune"),
           ("saturn","uranus"),("saturn","neptune"),("saturn","pluto"),
           ("uranus","neptune"),("uranus","pluto"),("neptune","pluto")]
BR_ASPECTS = {0: None, 60: +1, 90: -1, 120: +1, 180: -1}    # 60/120 +, 90/180 -

def _conj_sign(a, b):
    # Bradley Fig-7 valency table not public; heuristic: hard/malefic bodies negative.
    return -1 if ({"saturn","pluto","mars"} & {a, b}) else +1

def _potential(lonA, lonB, conj_sign):
    sep = abs(((lonA - lonB + 180) % 360) - 180)
    best = 0.0
    for ang, sgn in BR_ASPECTS.items():
        d = abs(sep - ang)
        if d <= 15:
            val = 10.0 * math.cos((d / 15.0) * (math.pi / 2))
            s = conj_sign if ang == 0 else sgn
            cand = s * val
            if abs(cand) > abs(best):
                best = cand
    return best

BODIES = {
    "sun": swe.SUN, "moon": swe.MOON, "mercury": swe.MERCURY, "venus": swe.VENUS,
    "mars": swe.MARS, "jupiter": swe.JUPITER, "saturn": swe.SATURN,
    "uranus": swe.URANUS, "neptune": swe.NEPTUNE, "pluto": swe.PLUTO,
    "node": swe.MEAN_NODE,
}
# classic "market" pairs (slow hard aspects + Mars/Sun stress aspects)
PAIRS = [
    ("jupiter", "saturn"), ("saturn", "uranus"), ("saturn", "neptune"),
    ("saturn", "pluto"), ("jupiter", "uranus"), ("jupiter", "neptune"),
    ("uranus", "pluto"), ("uranus", "neptune"), ("mars", "saturn"),
    ("mars", "uranus"), ("sun", "saturn"), ("sun", "uranus"), ("venus", "mars"),
]
HARD = [0.0, 90.0, 180.0]      # conjunction / square / opposition
ALL_ASPECTS = [0.0, 60.0, 90.0, 120.0, 180.0]

def jd_of(y, m, d, hour=21.0):
    return swe.julday(y, m, d, hour)

def _lon_speed(jd, ipl):
    xx, _ = swe.calc_ut(jd, ipl, FLAGS)
    return float(xx[0]), float(xx[3])

def sep_angle(a, b):
    """folded separation in [0,180]"""
    d = abs((a - b) % 360.0)
    return d if d <= 180.0 else 360.0 - d

def features_for(y, m, d):
    jd = jd_of(y, m, d)
    lon, spd = {}, {}
    for name, ipl in BODIES.items():
        lon[name], spd[name] = _lon_speed(jd, ipl)

    f = {}
    n_retro = 0
    for name in BODIES:
        r = math.radians(lon[name])
        f[f"ph_{name}_sin"] = math.sin(r)
        f[f"ph_{name}_cos"] = math.cos(r)
        f[f"ph_{name}_spd"] = spd[name]
        retro = 1 if spd[name] < 0 else 0
        f[f"ph_{name}_retro"] = retro
        if name not in ("sun", "moon", "node"):
            n_retro += retro
    f["ph_n_retro"] = n_retro
    f["ph_mercury_retro"] = 1 if spd["mercury"] < 0 else 0

    # lunar synodic phase: Moon - Sun angle (0=new, 180=full)
    phase = (lon["moon"] - lon["sun"]) % 360.0
    pr = math.radians(phase)
    f["ph_moon_phase_sin"] = math.sin(pr)
    f["ph_moon_phase_cos"] = math.cos(pr)
    f["ph_moon_illum"] = (1 - math.cos(pr)) / 2.0          # 0 new -> 1 full
    f["ph_moon_dist_new"] = min(phase, 360.0 - phase)       # deg to new moon
    f["ph_moon_dist_full"] = abs(phase - 180.0)             # deg to full moon

    # pairwise aspects
    pot2 = pot3 = 0.0
    for a, b in PAIRS:
        s = sep_angle(lon[a], lon[b])
        key = f"{a[:3]}_{b[:3]}"
        f[f"ph_sep_{key}"] = s
        f[f"ph_cos2_{key}"] = math.cos(math.radians(2 * s))
        # proximity to nearest hard aspect (1 at exact, decays over 8deg orb)
        orb = min(abs(s - A) for A in HARD)
        f[f"ph_hard_{key}"] = max(0.0, 1.0 - orb / 8.0)
    # global harmonic potentials over all body pairs (Bradley-ish)
    names = [n for n in BODIES if n != "node"]
    cnt = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            s = sep_angle(lon[names[i]], lon[names[j]])
            rad = math.radians(s)
            pot2 += math.cos(2 * rad)
            pot3 += math.cos(3 * rad)
            cnt += 1
    f["ph_aspect_pot2"] = pot2 / cnt
    f["ph_aspect_pot3"] = pot3 / cnt

    # ---- declinations (equatorial) ----
    decl = {}
    for nm in ("venus", "mars", "moon", "sun"):
        xe, _ = swe.calc_ut(jd, BODIES[nm], FLGE)
        decl[nm] = float(xe[1])
    f["ph_decl_venus"] = decl["venus"]
    f["ph_decl_mars"] = decl["mars"]
    f["ph_decl_moon"] = decl["moon"]
    f["ph_moon_decl_extreme"] = max(0.0, abs(decl["moon"]) - 18.0)   # high-decl lore

    # ---- Bradley Siderograph ----
    D = 0.5 * (decl["venus"] + decl["mars"])
    long_pairs = set(BR_LONG)
    L = sum(_potential(lon[a], lon[b], _conj_sign(a, b)) for a, b in BR_LONG)
    M = 0.0
    for i in range(len(BR_BODIES)):
        for j in range(i + 1, len(BR_BODIES)):
            a, b = BR_BODIES[i], BR_BODIES[j]
            if (a, b) in long_pairs or (b, a) in long_pairs:
                continue
            M += _potential(lon[a], lon[b], _conj_sign(a, b))
    f["ph_bradley_L"] = L
    f["ph_bradley_M"] = M
    f["ph_bradley_D"] = D
    f["ph_bradley_P"] = 5.0 * (L + D) + M

    # ---- heliocentric longitudes (Mercury/Venus/Earth) ----
    for nm in ("mercury", "venus", "earth"):
        ipl = swe.EARTH if nm == "earth" else BODIES[nm]
        xh, _ = swe.calc_ut(jd, ipl, FLGH)
        r = math.radians(float(xh[0]))
        f[f"ph_helio_{nm}_sin"] = math.sin(r)
        f[f"ph_helio_{nm}_cos"] = math.cos(r)
    return f

if __name__ == "__main__":
    import json
    print(json.dumps(features_for(2020, 2, 19), indent=2)[:800])
    print("n_features =", len(features_for(2020, 2, 19)))
