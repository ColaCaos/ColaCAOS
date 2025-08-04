import numpy as np
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, Slider, CustomJS
from bokeh.layouts import column, row
from bokeh.io import output_notebook

# Esto activa la inserción de JS/CSS de Bokeh en el HTML
output_notebook(hide_banner=True)

# Parámetros iniciales
r0, x0_0, steps = 3.2, 0.2, 100

def logistic_series(r, x0, steps):
    xs = np.linspace(0,1,200)
    ys = r*xs*(1-xs)
    series = [x0]
    x = x0
    for _ in range(steps):
        x = r*x*(1-x)
        series.append(x)
    xc, yc = [series[0]], [series[0]]
    for val in series[1:]:
        xc.append(xc[-1]); yc.append(val)
        xc.append(val);    yc.append(val)
    return xs, ys, xc, yc, series

# Datos iniciales
xs, ys, xc, yc, series = logistic_series(r0, x0_0, steps)

# Fuentes de datos
src_curve    = ColumnDataSource(data=dict(x=xs,      y=ys))
src_identity = ColumnDataSource(data=dict(x=xs,      y=xs))
src_cobweb   = ColumnDataSource(data=dict(x=xc,      y=yc))
src_series   = ColumnDataSource(data=dict(x=list(range(len(series))), y=series))

# Figuras
p1 = figure(width=400, height=400, title="Cobweb")
p1.line('x','y', source=src_curve,    legend_label='f(x)')
p1.line('x','y', source=src_identity, line_dash='dashed', legend_label='y=x')
p1.line('x','y', source=src_cobweb,   line_color='red', legend_label='Cobweb')

p2 = figure(width=400, height=400, title="Evolución de xₙ")
p2.line('x','y', source=src_series)
p2.scatter('x','y', source=src_series, size=6)

# Sliders
slider_r  = Slider(start=0.5, end=4.0, step=0.05, value=r0,  title="r")
slider_x0 = Slider(start=0.0, end=1.0, step=0.05, value=x0_0, title="x₀")

# Callback en JS
callback = CustomJS(args=dict(
    src_curve=src_curve, src_cobweb=src_cobweb, src_series=src_series,
    slider_r=slider_r, slider_x0=slider_x0, steps=steps
), code="""
  function logistic_series(r, x0, steps) {
    let xs = [], ys = [];
    for(let i=0;i<200;i++){
      let x = i/199;
      xs.push(x);
      ys.push(r*x*(1-x));
    }
    let series=[x0], x=x0;
    for(let i=0;i<steps;i++){
      x = r*x*(1-x);
      series.push(x);
    }
    let xc=[series[0]], yc=[series[0]];
    for(let i=1;i<series.length;i++){
      let v = series[i];
      xc.push(xc[xc.length-1]); yc.push(v);
      xc.push(v);            yc.push(v);
    }
    return {xs, ys, xc, yc, series};
  }
  const r = slider_r.value;
  const x0 = slider_x0.value;
  const res = logistic_series(r, x0, steps);
  src_curve.data   = {x: res.xs, y: res.ys};
  src_cobweb.data  = {x: res.xc, y: res.yc};
  src_series.data = {x: res.series.map((_,i)=>i), y: res.series};
  src_curve.change.emit();
  src_cobweb.change.emit();
  src_series.change.emit();
""");

slider_r.js_on_change('value', callback);
slider_x0.js_on_change('value', callback);

layout = column(
  row(p1, p2),
  row(slider_r, slider_x0)
)

show(layout)
