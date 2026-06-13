# 변동성 Skew & 하락 선행지표 연구 노트

> SPY 옵션(2019–2026)으로 변동성 skew를 직접 계산하고, "발동하면 향후 1개월 지수 하락 확률이 높아지는" **선행지표**를 찾아간 과정 정리.
> (백테스트·차트 구현 부분은 제외. 개념과 지표 탐색 위주.)

---

## 1. 기본 개념

### 1.1 Volatility Skew란
같은 만기의 옵션이라도 **행사가(strike)마다 내재변동성(IV)이 다른 현상**. Black-Scholes 가정대로면 모든 strike의 IV가 같아야 하지만, 실제로는 다름.

- **주식/지수 skew(smirk, 스미르크)**: OTM **풋**의 IV가 OTM **콜**보다 높다. 투자자가 폭락을 더 두려워해 하방 보호(풋)에 프리미엄을 지불 → "빠르게 빠지고 천천히 오른다".
- skew는 **분포의 비대칭(왼쪽 꼬리)**을 가격으로 표현한 것.

### 1.2 절대 skew vs 정규화 skew — ★중요
| | 정의 | 의미 |
|---|---|---|
| **절대 skew** | IV(25Δ Put) − IV(25Δ Call) | vol 레벨을 따라감. 폭락 때 모든 IV가 높아져 같이 폭발 |
| **정규화 skew** | (IV₂₅ₚ − IV₂₅꜀) / IV_ATM | **모양(shape)만 분리**. 스케일 무관, 시점 비교 가능 |

실측: 2020-03 코로나 폭락 때 **절대** skew는 26.8 vol포인트로 폭발했지만 **정규화** skew(0.36)는 폭락 *전* 고점(2020-02, 0.36)과 거의 같았다. → **절대 skew = vol 레벨, 정규화 skew = 비대칭의 모양.**

### 1.3 25-delta skew 계산법 (개념)
업계 표준 "synthetic 30-DTE 25Δ skew":
1. **Delta targeting**: 고정 moneyness 대신 **델타 ≈ ±0.25**인 OTM 풋·콜을 선택 (델타가 strike·만기·변동성을 모두 반영해 더 robust).
2. **Synthetic 30-DTE**: 30일을 끼는 두 만기를 **total-variance 공간**(V = IV²·t)에서 보간 → 일정한 30일 기준점.
3. **ATM IV**: ±0.5 델타 콜·풋 IV 평균.
4. `25Δ Skew = IV₂₅ₚ − IV₂₅꜀`, `Normalized = 그것 / IV_ATM`.

### 1.4 Term Structure (기간구조)
**단기 IV vs 장기 IV**의 관계.
- **Contango(정상)**: 단기 < 장기 (`ts_ratio < 1`). 평온.
- **Backwardation/역전(스트레스)**: 단기 > 장기 (`ts_ratio > 1`). 근접 위험이 가격에 반영됨. (VIX/VIX3M > 1 과 같은 개념)
- `ts_ratio = 단기 ATM IV ÷ 장기 ATM IV`.

### 1.5 VRP (Variance/Volatility Risk Premium)
`VRP = 내재변동성(IV) − 실제 실현변동성(RV)`. 구조적으로 **+**(IV가 RV보다 비쌈) → 옵션 *매도*가 장기적으로 캐리를 먹는 이유.
- 실측: VRP 중앙값 **+1.97 vol포인트, 68% 양수**. 평균이 중앙값보다 훨씬 작음 = **fat 좌측꼬리**(폭락 때 RV가 IV 추월) → "대체로 먹다가 가끔 크게 토함".
- 옵션 *롱*(콜/풋 매수)은 이 VRP를 **지불하는** 쪽 = 구조적 역풍.

### 1.6 Index skew vs Single-name skew (dispersion / correlation)
- **지수 skew는 개별종목 skew보다 항상 가파르다.** 이유 = **상관관계(correlation)**. 폭락 땐 종목들이 같이 떨어져(corr→1) 분산효과가 사라지므로 지수 하방 풋 수요가 구조적으로 강함.
- 이 차이를 먹는 게 **dispersion / correlation 트레이드**.

---

## 2. 핵심 질문: skew가 선행지표가 될 수 있나?

### 2.1 가설
"skew가 가팔라지면(steepening) → 향후 지수가 하락/급락한다" → 풋을 사서 헤지.

### 2.2 실증 결과: **skew 단독은 방향 예측 못 함**
**다음 1주일(5거래일) 결과 기준:**

| | corr(Δskew, fwd 5d) |
|---|---|
| 정규화 skew 변화 | **−0.016** (≈0) |
| 절대 skew 변화 | **−0.067** (거의 0) |

- 가장 가팔라진 상위 분위(Q5)의 다음주 하락확률 ≈ **base rate와 동일**. 엣지 없음.
- **포워드 수익률 상관**: 정규화 skew **+0.005**(무관), 절대 skew **+0.246**, ATM IV **+0.309**.

**왜?**
1. skew 가팔라짐은 선행이 아니라 **반응/동행** — 흔들리면 풋이 사지면서 *동시에* 가팔라짐.
2. **리드타임이 길고 들쭉날쭉** — 2021년은 skew가 1년 내내 가파르다가 2022에 터짐. 1주 창엔 안 잡힘.
3. **가팔라짐 ≠ 약세** — 멜트업 때 *콜이 죽어서*도 가팔라짐(방향 오염).
4. → 학계 통설과 일치: **옵션 skew는 지수 단기 타이밍엔 예측력이 약함.**

---

## 3. 선행지표 탐색 과정

방향(위/아래)이 아니라 **"향후 1개월 하락 위험을 base rate보다 유의미하게 올리는 지표"**(=헤지 트리거)로 목표를 재정의. 여러 지표를 일제히 스크리닝.

**측정 대상**: `fwd_min_21` = 향후 21거래일 내 최악 낙폭.
**Base rate**: P(−5% 이내) **18%**, P(−10% 이내) **4.6%**, 평균 최악낙폭 **−2.93%**.

### 3.1 지표별 위험구역 → 향후 1개월 하락위험

| 지표 (danger zone) | P(−5%) | P(−10%) | 판정 |
|---|---|---|---|
| **belowMA + IV%ile≥60 (조합)** | **40%** | **17.5%** | 최강 (~3.8× base) |
| below 100일선 (하락추세) | 35% | 15.3% | 강함 |
| IV %ile ≥80 (고변동성) | 38% | 15.1% | 강함 |
| skew %ile ≤20 (평평/방심) | 27% | 7.0% | 약 |
| VRP < 0 (실현>내재) | 23% | 8.2% | 약 |
| **skew %ile ≥80 (가파름)** | **16%** | 5.9% | **≈base = 신호 아님** |
| IV %ile ≤20 (저변동성) | 8% | 0.0% | 안전(평온) |
| 저IV + 가파른skew (2021형) | 5% | 0.0% | 안전(단, 리드타임 김) |

### 3.2 Term structure 역전 추가 (가장 부합한 greeks 지표)

| 지표 | P(−5%) | P(−10%) | 판정 |
|---|---|---|---|
| **ts%ile≥80 + belowMA** | **44%** | **20.3%** | 최강 (~4.4× base) |
| ts_ratio > 1.05 (강한 역전) | 32% | 15.0% | 강함 (~3.3× base) |
| ts_ratio > 1.0 (역전) | 29% | 9.6% | 중 |
| **ts%ile ≤20 (가파른 contango)** | 9% | **0.4%** | "헤지 불필요" 신호 |

### 3.3 핵심 결론
1. **skew(모양)는 하락 선행지표가 아니다.** 가파른 skew의 하락확률 = base rate. 오히려 *평평한* skew가 약간 더 위험(방심/하락 진행 중).
2. **단일 최강 트리거는 ① 추세(100일선 아래) ② IV 레벨(고변동성) ③ term-structure 역전.** 셋 다 −10% 확률을 base 4.6% → ~15%로 **3배**.
3. **양방향 판별력**: 가파른 contango / 저IV는 하락확률을 0%에 가깝게 → **"헤지 해제" 신호**로도 유용.
4. greeks 연관성은 맞지만, 정보를 담은 건 **skew(스큐)가 아니라 vol의 level·term(기간구조)**.

### 3.4 "급락 직전" 평균 상태
| | 급락(향후 −10%) 직전 | 평소 |
|---|---|---|
| 정규화 skew | 0.353 | 0.340 (거의 차이 없음) |
| ATM IV | **21.5%** | 17.0% |

→ 급락 직전엔 **skew가 아니라 IV가 먼저 올라와 있다.**

---

## 4. McElligott(Nomura) 프레임워크 — 왜 그는 skew를 중요하게 보나

전사본/노트 종합: 그는 skew를 **방향 예측기가 아니라 "시스템 상태·취약성 지도"**로 쓴다. 우리 실증과 정확히 일치.

### 4.1 "We can't crash until skew is steep" ★핵심
> 가파른 skew = 다들 헤지됨 = **딜러가 하방을 숏(short gamma)** → 하락 시작되면 딜러 헤지가 **가속(accelerant flow)** → 크래시로 증폭.

즉 **steep skew = 크래시의 *필요조건/연료*이지 *방아쇠*가 아님.** 점화엔 **catalyst**(그가 꼽는 건 "nasty NFP print" 같은 노동 데이터, 소비자 균열)가 필요. → 우리가 "skew 단독 타이밍 안 됨"을 확인한 이유.

### 4.2 그의 *실제* 1순위 = 포지셔닝/레버리지 극단
- gross exposure 퍼센타일(risk parity 99.7%ile, 헤지펀드 gross 100%ile, 넷 숏달러 0%ile 등)이 극단 + 가격이 "bending off the curve"(포물선) + 내러티브 정체 → **profit-taking이 강제 디그로싱으로 전환**.
- 크라우딩된 컨센서스를 찾고, 그게 풀릴 때의 *기계적 플로우*를 본다. "anticipating the anticipators."

### 4.3 spot-up-vol-up
spot↑ 와 vol↑ 가 동시에 = **강제 상방 추격 멜트업** → "collapses under the weight of its own delta". 천장 근처 신호.

### 4.4 그가 보는 지표들 (우선순위)
1. **포지셔닝/gross %ile** (1순위)
2. **딜러 감마/vanna** (숏감마 = accelerant, 롱감마/buyback = shock absorber)
3. **Skew** (fragility 게이지, 방향 아님)
4. **Vol surface**: realized vs implied(VRP), VIX, vol-of-vol, **term structure**, spot-up-vol-up
5. **Correlation/Dispersion** (지수 vs 개별종목)
6. **구조적 플로우**: buyback("15년 최대 수요, synthetic long gamma, vol suppressor"), leveraged ETF(synthetic neg gamma), vol-control/CTA(VAR=모멘텀), 0DTE, 프리미엄 인컴 ETF
7. **Catalyst**: 노동(NFP)·소비자·Fed·credit spread

---

## 5. 결론 / 실전 함의

### 5.1 선행 하락지표 = **2단계 (연료 × 점화)**
단일 숫자가 아니라 조건부:
- **연료(precondition)**: `skew %ile 높음` → 딜러 숏다운사이드, accelerant 장전
- **점화(trigger)**: `term structure 역전(ts_ratio>1.05)` 또는 `100일선 하향` 또는 `IV 상승 전환`
- **헤지 ON**: 연료+점화 동시. **OFF**: contango 복귀 + skew 정상화.

### 5.2 실제 사례 검증 — 2026-02 크래시 (McElligott가 예고했던 그 국면)
SPX 6979(1/27) → **6344(3/30, −9%)**. 우리 지표 추적:
- **연료**: 1월 중순~2월 초 **skew %ile 90~100%ile** 도달 (딜러 숏다운사이드 적재)
- **점화**: 2/4~2/5 **term structure 역전 시작 + IV 상승**, SPX 아직 6800대 (헤지 진입 적기)
- **폭락**: 2/12~3/9 skew 100%ile + 깊은 역전(ts 1.13~1.25) + IV 급등

### 5.3 한계 (정직하게)
- **점화 신호도 약간 늦/동행** — term structure가 깊이 역전·IV 90%ile 될 땐 이미 하락 진행 중. 가장 이른 경고(skew 100%ile)는 노이즈가 큼(몇 주 유지). → **"싸고 이른 확실한 경고"는 존재하면 차익거래돼 사라짐.**
- **표본**: SPY 프록시, 불장 위주(2019–2026), −10% 이벤트 ~50개뿐 → 통계적 검정력 낮음.
- McElligott의 진짜 1순위(**포지셔닝/gross %ile, 딜러 감마 과거 시계열**)는 무료 데이터로 재현 불가 → 우리 지표가 "fragility는 보되 타이밍은 약한" 한계가 그의 프레임에서도 설명됨.

### 5.4 한 줄 요약
> **Skew는 "신호(signal)"가 아니라 "상태(context)"다.** 크래시의 연료를 측정하지, 시점을 맞히지 않는다. 실전 하락 트리거는 **추세 + IV 레벨 + term-structure 역전**의 조합이며, skew는 그 페이오프를 증폭시키는 *조건 변수*다.

---

## 부록: 데이터 소스
| 용도 | 소스 | 비고 |
|---|---|---|
| 과거 옵션 체인(IV·delta) | DoltHub `post-no-preference/options`, **SPY** | 2019-02~2026-06, 일부 subsampling(2020–23 주 3일), SPX 없음 |
| 지수 종가 | FRED `SP500` | 키 불필요 |
| 일봉 OHLC | Yahoo `^GSPC` | 캔들용 |
| 실시간 체인(forward) | CBOE `delayed_quotes` `_SPX`/`_VIX` | 오늘 스냅샷만 |

*참고 레퍼런스: github.com/anthonymakarewicz/volatility-trading (25Δ skew 방법론·VRP·skew 평균회귀 — 코드만, 데이터 없음).*
