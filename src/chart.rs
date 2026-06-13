//! Self-contained HTML chart (server-side SVG + a small inline hover script).
//! No external libraries, no CDN — opens in any browser offline.

pub struct Joined {
    pub date: String,
    pub close: f64,
    pub drawdown: f64,
    pub skew_abs: f64,
    pub skew_norm: f64,
    pub atm_iv: f64,
    pub vrp: f64, // ATM IV - trailing 21D realized vol (vol points, ex-ante)
}

const W: f64 = 1280.0;
const ML: f64 = 64.0; // left margin
const MR: f64 = 24.0; // right margin

struct Panel {
    top: f64,
    h: f64,
}

fn x_of(i: usize, n: usize) -> f64 {
    let plotw = W - ML - MR;
    if n <= 1 {
        ML
    } else {
        ML + (i as f64) / ((n - 1) as f64) * plotw
    }
}

/// Build a polyline `points` string for a value accessor, mapping [vmin,vmax]->[bottom,top].
fn polyline(rows: &[Joined], p: &Panel, vmin: f64, vmax: f64, log: bool, f: impl Fn(&Joined) -> f64) -> String {
    let n = rows.len();
    let (lo, hi) = if log { (vmin.ln(), vmax.ln()) } else { (vmin, vmax) };
    let span = if (hi - lo).abs() < 1e-12 { 1.0 } else { hi - lo };
    let mut s = String::new();
    for (i, r) in rows.iter().enumerate() {
        let mut v = f(r);
        if !v.is_finite() {
            continue;
        }
        if log {
            v = v.max(1e-9).ln();
        }
        let y = p.top + p.h - (v - lo) / span * p.h;
        if !s.is_empty() {
            s.push(' ');
        }
        s.push_str(&format!("{:.1},{:.1}", x_of(i, n), y));
    }
    s
}

fn minmax(rows: &[Joined], f: impl Fn(&Joined) -> f64) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for r in rows {
        let v = f(r);
        if v.is_finite() {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    if !lo.is_finite() {
        (0.0, 1.0)
    } else {
        (lo, hi)
    }
}

/// Vertical shaded bands where drawdown <= threshold (e.g. -0.10 for >=10% off-highs).
fn drawdown_bands(rows: &[Joined], threshold: f64, panels: &[&Panel]) -> String {
    let n = rows.len();
    let mut s = String::new();
    let mut i = 0;
    while i < n {
        if rows[i].drawdown <= threshold {
            let start = i;
            while i < n && rows[i].drawdown <= threshold {
                i += 1;
            }
            let x0 = x_of(start, n);
            let x1 = x_of(i.saturating_sub(1), n);
            for p in panels {
                s.push_str(&format!(
                    "<rect x='{:.1}' y='{:.1}' width='{:.1}' height='{:.1}' fill='#ff3b30' opacity='0.08'/>",
                    x0, p.top, (x1 - x0).max(1.0), p.h
                ));
            }
        } else {
            i += 1;
        }
    }
    s
}

fn year_gridlines(rows: &[Joined], panels: &[&Panel], bottom: f64) -> String {
    let n = rows.len();
    let mut s = String::new();
    let mut last_year = String::new();
    for (i, r) in rows.iter().enumerate() {
        let year = r.date.get(0..4).unwrap_or("").to_string();
        if year != last_year {
            last_year = year.clone();
            let x = x_of(i, n);
            for p in panels {
                s.push_str(&format!(
                    "<line x1='{:.1}' y1='{:.1}' x2='{:.1}' y2='{:.1}' stroke='#e5e5e5' stroke-width='1'/>",
                    x, p.top, x, p.top + p.h
                ));
            }
            s.push_str(&format!(
                "<text x='{:.1}' y='{:.1}' font-size='11' fill='#888' text-anchor='middle'>{}</text>",
                x, bottom + 16.0, year
            ));
        }
    }
    s
}

fn axis_labels(p: &Panel, vmin: f64, vmax: f64, fmt: impl Fn(f64) -> String, ticks: usize) -> String {
    let mut s = String::new();
    for k in 0..=ticks {
        let frac = k as f64 / ticks as f64;
        let v = vmin + (vmax - vmin) * frac;
        let y = p.top + p.h - frac * p.h;
        s.push_str(&format!(
            "<line x1='{:.1}' y1='{:.1}' x2='{:.1}' y2='{:.1}' stroke='#f0f0f0'/>\
             <text x='{:.1}' y='{:.1}' font-size='10' fill='#999' text-anchor='end'>{}</text>",
            ML, y, W - MR, y, ML - 6.0, y + 3.0, fmt(v)
        ));
    }
    s
}

fn panel_title(p: &Panel, t: &str) -> String {
    format!(
        "<text x='{:.1}' y='{:.1}' font-size='12' font-weight='600' fill='#333'>{}</text>",
        ML, p.top - 6.0, t
    )
}

/// Interactive Plotly dashboard: 6 stacked, x-synced panels (zoom / pan / unified
/// hover / rangeslider + range buttons). Includes PDF-style rolling percentile panels.
pub fn write_dashboard(path: &str, title: &str, rows: &[Joined]) -> std::io::Result<()> {
    // Build JS array literals (null for non-finite).
    let num_arr = |f: &dyn Fn(&Joined) -> f64, scale: f64| -> String {
        let mut s = String::from("[");
        for (i, r) in rows.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            let v = f(r) * scale;
            if v.is_finite() {
                s.push_str(&format!("{:.4}", v));
            } else {
                s.push_str("null");
            }
        }
        s.push(']');
        s
    };
    let mut dates = String::from("[");
    for (i, r) in rows.iter().enumerate() {
        if i > 0 {
            dates.push(',');
        }
        dates.push('"');
        dates.push_str(&r.date);
        dates.push('"');
    }
    dates.push(']');

    let data_js = format!(
        "const TITLE={};\nconst DATES={};\nconst CLOSE={};\nconst ATM={};\nconst SKEWN={};\nconst SKEWA={};\nconst VRP={};\n",
        serde_json::to_string(title).unwrap(),
        dates,
        num_arr(&|r| r.close, 1.0),
        num_arr(&|r| r.atm_iv, 100.0),
        num_arr(&|r| r.skew_norm, 1.0),
        num_arr(&|r| r.skew_abs, 100.0),
        num_arr(&|r| r.vrp, 100.0),
    );

    let head = r#"<!DOCTYPE html><html><head><meta charset="utf-8">
<title>skew dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
<style>body{margin:0;font-family:-apple-system,Segoe UI,Roboto,sans-serif}#chart{width:100%}</style>
</head><body><div id="chart"></div><script>
"#;

    // Static JS: computes rolling percentiles and builds the figure. No format! here
    // (raw braces), so nothing needs escaping.
    let body = r#"
function rollpct(a,w){const o=new Array(a.length).fill(null);
  for(let i=0;i<a.length;i++){if(a[i]==null)continue;const lo=Math.max(0,i-w+1);let c=0,t=0;
    for(let j=lo;j<=i;j++){if(a[j]==null)continue;t++;if(a[j]<=a[i])c++;}
    o[i]=t>1?Math.round(c/t*100):null;}
  return o;}
const W=252; // ~1yr of observations (series is subsampled, so this is approximate)
const IVP=rollpct(ATM,W), SKP=rollpct(SKEWN,W);
const C={width:1};
const traces=[
 {x:DATES,y:CLOSE,name:'S&P 500',yaxis:'y',line:{color:'#1f77b4',width:1},hovertemplate:'%{y:.0f}<extra>SPX</extra>'},
 {x:DATES,y:ATM,name:'ATM IV %',yaxis:'y2',line:{color:'#7f3fbf',width:1},hovertemplate:'%{y:.1f}%<extra>ATM IV</extra>'},
 {x:DATES,y:IVP,name:'IV %ile (1Y)',yaxis:'y3',line:{color:'#7f3fbf',width:1},fill:'tozeroy',fillcolor:'rgba(127,63,191,.10)',hovertemplate:'%{y:.0f}%ile<extra>IV pctile</extra>'},
 {x:DATES,y:SKEWN,name:'25Δ skew (norm)',yaxis:'y4',line:{color:'#d62728',width:1},hovertemplate:'%{y:.3f}<extra>skew norm</extra>'},
 {x:DATES,y:SKP,name:'skew %ile (1Y)',yaxis:'y5',line:{color:'#d62728',width:1},fill:'tozeroy',fillcolor:'rgba(214,39,40,.10)',hovertemplate:'%{y:.0f}%ile<extra>skew pctile</extra>'},
 {x:DATES,y:VRP,name:'VRP = IV−RV (pts)',yaxis:'y6',line:{color:'#2ca02c',width:1},hovertemplate:'%{y:.1f}pt<extra>VRP</extra>'},
];
const layout={
 title:{text:TITLE,font:{size:15}}, height:1500, hovermode:'x unified',
 legend:{orientation:'h',y:1.02,x:0}, margin:{l:64,r:20,t:60,b:40},
 xaxis:{domain:[0,1],anchor:'y6',type:'date',
   rangeslider:{visible:true,thickness:0.035},
   rangeselector:{buttons:[{step:'month',count:1,label:'1m'},{step:'month',count:6,label:'6m'},{step:'year',count:1,label:'YTD',stepmode:'todate'},{step:'year',count:1,label:'1y'},{step:'all',label:'all'}]}},
 yaxis: {domain:[0.86,1.00],type:'log',title:'SPX (log)'},
 yaxis2:{domain:[0.70,0.835],title:'ATM IV %'},
 yaxis3:{domain:[0.555,0.685],title:'IV %ile',range:[0,100]},
 yaxis4:{domain:[0.410,0.540],title:'skew norm'},
 yaxis5:{domain:[0.265,0.395],title:'skew %ile',range:[0,100]},
 yaxis6:{domain:[0.080,0.250],title:'VRP pts',zeroline:true,zerolinecolor:'#bbb'},
 shapes:[{type:'line',xref:'paper',x0:0,x1:1,yref:'y6',y0:0,y1:0,line:{color:'#bbb',width:1}}],
};
Plotly.newPlot('chart',traces,layout,{responsive:true,scrollZoom:true});
</script></body></html>"#;

    let mut html = String::with_capacity(head.len() + data_js.len() + body.len());
    html.push_str(head);
    html.push_str(&data_js);
    html.push_str(body);
    std::fs::write(path, html)
}

pub fn write_html(path: &str, title: &str, rows: &[Joined]) -> std::io::Result<()> {
    let p_price = Panel { top: 40.0, h: 240.0 };
    let p_norm = Panel { top: 330.0, h: 150.0 };
    let p_atm = Panel { top: 530.0, h: 150.0 };
    let height = 720.0;

    let (pc_lo, pc_hi) = minmax(rows, |r| r.close);
    let (sn_lo, sn_hi) = minmax(rows, |r| r.skew_norm);
    let (atm_lo, atm_hi) = minmax(rows, |r| r.atm_iv);

    let bands = drawdown_bands(rows, -0.10, &[&p_price, &p_norm, &p_atm]);
    let grid = year_gridlines(rows, &[&p_price, &p_norm, &p_atm], p_atm.top + p_atm.h);

    let price_line = polyline(rows, &p_price, pc_lo, pc_hi, true, |r| r.close);
    let norm_line = polyline(rows, &p_norm, sn_lo, sn_hi, false, |r| r.skew_norm);
    let atm_line = polyline(rows, &p_atm, atm_lo, atm_hi, false, |r| r.atm_iv);

    // inline data for the hover layer (emit JSON null for non-finite values)
    let jf = |v: f64, dp: usize| -> String {
        if v.is_finite() {
            format!("{:.*}", dp, v)
        } else {
            "null".to_string()
        }
    };
    let data_json: String = {
        let mut s = String::from("[");
        for (i, r) in rows.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&format!(
                "{{\"d\":\"{}\",\"c\":{},\"dd\":{},\"sa\":{},\"sn\":{},\"a\":{}}}",
                r.date,
                jf(r.close, 2),
                jf(r.drawdown, 3),
                jf(r.skew_abs, 4),
                jf(r.skew_norm, 4),
                jf(r.atm_iv, 4)
            ));
        }
        s.push(']');
        s
    };

    let html = format!(
        r##"<!DOCTYPE html><html><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:16px;color:#222}}
.wrap{{position:relative;width:{w}px}} #tip{{position:absolute;pointer-events:none;background:#111;color:#fff;
font-size:11px;padding:6px 8px;border-radius:4px;opacity:0;white-space:pre;transform:translate(8px,8px);z-index:10}}
h2{{font-size:16px;margin:0 0 4px}} .sub{{color:#777;font-size:12px;margin:0 0 12px}}</style></head>
<body>
<h2>{title}</h2>
<p class="sub">Red bands = SPX ≥10% below its running high (drawdowns). Top: SPX (log). Middle: normalized 25Δ skew = (IV<sub>25P</sub>−IV<sub>25C</sub>)/IV<sub>ATM</sub>. Bottom: ATM IV (vol level). Hover for values.</p>
<div class="wrap">
<svg id="svg" width="{w}" height="{h}" style="background:#fff">
{bands}
{grid}
{price_grid}
{norm_grid}
{atm_grid}
{ptitle}
{ntitle}
{atitle}
<polyline fill="none" stroke="#1f77b4" stroke-width="1.2" points="{price_line}"/>
<polyline fill="none" stroke="#d62728" stroke-width="1.2" points="{norm_line}"/>
<polyline fill="none" stroke="#7f3fbf" stroke-width="1.2" points="{atm_line}"/>
<line id="cross" x1="0" y1="40" x2="0" y2="680" stroke="#999" stroke-dasharray="3,3" style="opacity:0"/>
</svg>
<div id="tip"></div>
</div>
<script>
const DATA={data_json};
const ML={ml}, MR={mr}, W={w};
const svg=document.getElementById('svg'), tip=document.getElementById('tip'), cross=document.getElementById('cross');
const plotw=W-ML-MR, n=DATA.length;
svg.addEventListener('mousemove',e=>{{
  const rect=svg.getBoundingClientRect(); const mx=e.clientX-rect.left;
  let i=Math.round((mx-ML)/plotw*(n-1)); if(i<0)i=0; if(i>=n)i=n-1;
  const x=ML+i/(n-1)*plotw; cross.setAttribute('x1',x); cross.setAttribute('x2',x); cross.style.opacity=1;
  const r=DATA[i]; const fx=(v,d)=>v==null?'–':v.toFixed(d);
  tip.style.opacity=1; tip.style.left=mx+'px'; tip.style.top=(e.clientY-rect.top)+'px';
  tip.textContent=`${{r.d}}\nSPX  ${{fx(r.c,2)}}  (dd ${{fx(r.dd*100,1)}}%)\nskew norm ${{fx(r.sn,3)}}  abs ${{fx(r.sa*100,2)}}pt\nATM IV ${{fx(r.a*100,1)}}%`;
}});
svg.addEventListener('mouseleave',()=>{{tip.style.opacity=0;cross.style.opacity=0;}});
</script>
</body></html>"##,
        title = title,
        w = W,
        h = height,
        ml = ML,
        mr = MR,
        bands = bands,
        grid = grid,
        price_grid = axis_labels(&p_price, pc_lo, pc_hi, |v| format!("{:.0}", v), 4),
        norm_grid = axis_labels(&p_norm, sn_lo, sn_hi, |v| format!("{:.2}", v), 4),
        atm_grid = axis_labels(&p_atm, atm_lo, atm_hi, |v| format!("{:.0}%", v * 100.0), 4),
        ptitle = panel_title(&p_price, "S&P 500 (log scale)"),
        ntitle = panel_title(&p_norm, "Normalized 25-delta skew  (IV_25P - IV_25C) / IV_ATM"),
        atitle = panel_title(&p_atm, "ATM implied vol (30D synthetic)"),
        price_line = price_line,
        norm_line = norm_line,
        atm_line = atm_line,
        data_json = data_json,
    );

    std::fs::write(path, html)
}
