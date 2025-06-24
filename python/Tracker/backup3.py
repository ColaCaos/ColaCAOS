# Autor: Rubén Torre Merino
import cv2
import numpy as np
import time
import csv
import matplotlib.pyplot as plt

def find_centroid(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    c = max(cnts, key=cv2.contourArea)
    M = cv2.moments(c)
    if M['m00'] == 0:
        return None, None
    return int(M['m10']/M['m00']), int(M['m01']/M['m00'])

def fill_nans(xs, ys):
    x = np.array(xs, dtype=np.float64)
    y = np.array(ys, dtype=np.float64)
    idx = np.arange(len(x))
    good = ~np.isnan(x)
    if good.sum()>1:
        x[~good] = np.interp(idx[~good], idx[good], x[good])
    good = ~np.isnan(y)
    if good.sum()>1:
        y[~good] = np.interp(idx[~good], idx[good], y[good])
    return x.tolist(), y.tolist()

def smooth(v, w=7):
    if len(v) < w:
        return v
    c = np.cumsum(np.insert(v, 0, 0))
    out = ((c[w:] - c[:-w]) / w).tolist()
    return out + v[-(w-1):]

def main():
    L1_cm, L2_cm = 23.5, 20.0
    max_reach = L1_cm + L2_cm

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 60)
    gpu_frame = cv2.cuda_GpuMat()

    # HSV sliders iniciales
    cv2.namedWindow("Ajustes", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("R_LO","Ajustes",  0,180,lambda x:None)
    cv2.createTrackbar("R_HI","Ajustes", 10,180,lambda x:None)
    cv2.createTrackbar("R2_LO","Ajustes",160,180,lambda x:None)
    cv2.createTrackbar("R2_HI","Ajustes",180,180,lambda x:None)
    cv2.createTrackbar("G_LO","Ajustes", 34,180,lambda x:None)
    cv2.createTrackbar("G_HI","Ajustes", 62,180,lambda x:None)
    cv2.createTrackbar("B_LO","Ajustes",100,180,lambda x:None)
    cv2.createTrackbar("B_HI","Ajustes",140,180,lambda x:None)
    cv2.createTrackbar("S_MIN","Ajustes", 88,255,lambda x:None)
    cv2.createTrackbar("V_MIN","Ajustes", 30,255,lambda x:None)

    # CSV
    csv_f = open('coordenadas_cm.csv','w',newline='')
    writer = csv.writer(csv_f)
    writer.writerow(['t_s',
                     'red_x_cm','red_y_cm',
                     'green_x_cm','green_y_cm',
                     'blue_x_cm','blue_y_cm'])

    # ventanas máscaras
    cv2.namedWindow("Mask Red",   cv2.WINDOW_NORMAL)
    cv2.namedWindow("Mask Green", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Mask Blue",  cv2.WINDOW_NORMAL)

    def detect():
        ret, frame = cap.read()
        if not ret:
            return None,None,None,None
        gpu_frame.upload(frame)
        hsv = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2HSV)
        s_min = cv2.getTrackbarPos("S_MIN","Ajustes")
        v_min = cv2.getTrackbarPos("V_MIN","Ajustes")
        masks = {}
        for color in ('red','green','blue'):
            acc = None
            if color=='red':
                lo1 = (cv2.getTrackbarPos("R_LO","Ajustes"), s_min, v_min)
                hi1 = (cv2.getTrackbarPos("R_HI","Ajustes"),255,255)
                lo2 = (cv2.getTrackbarPos("R2_LO","Ajustes"), s_min, v_min)
                hi2 = (cv2.getTrackbarPos("R2_HI","Ajustes"),255,255)
                m1 = cv2.cuda.inRange(hsv, lo1, hi1)
                m2 = cv2.cuda.inRange(hsv, lo2, hi2)
                acc = cv2.cuda.bitwise_or(m1, m2)
            else:
                lo = (cv2.getTrackbarPos(f"{color[0].upper()}_LO","Ajustes"), s_min, v_min)
                hi = (cv2.getTrackbarPos(f"{color[0].upper()}_HI","Ajustes"),255,255)
                acc = cv2.cuda.inRange(hsv, lo, hi)
            morph = cv2.cuda.createMorphologyFilter(
                cv2.MORPH_OPEN, cv2.CV_8UC1,
                cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
            )
            mask = morph.apply(acc).download()
            masks[color] = mask
        cr = find_centroid(masks['red'])
        cg = find_centroid(masks['green'])
        cb = find_centroid(masks['blue'])
        cv2.imshow("Mask Red",   masks['red'])
        cv2.imshow("Mask Green", masks['green'])
        cv2.imshow("Mask Blue",  masks['blue'])
        return cr, cg, cb, frame

    # calibración 2s
    d1s, d2s = [], []
    t0 = time.time()
    while time.time()-t0 < 2.0:
        cr, cg, cb, _ = detect()
        if None not in (*cr,*cg,*cb):
            d1 = np.hypot(cg[0]-cb[0], cg[1]-cb[1])
            d2 = np.hypot(cr[0]-cg[0], cr[1]-cg[1])
            if d1>5 and d2>5:
                d1s.append(d1); d2s.append(d2)
        if cv2.waitKey(1)&0xFF==ord('q'):
            cap.release(); csv_f.close(); return
    if not d1s or not d2s:
        print("Falló calibración")
        cap.release(); csv_f.close(); return
    s1 = L1_cm/np.mean(d1s)
    s2 = L2_cm/np.mean(d2s)
    print(f"Calibrado: s1={s1:.3f}, s2={s2:.3f}")

    # plot
    plt.ion()
    fig, ax = plt.subplots()
    lr, = ax.plot([], [], 'r-', label='Rojo')
    lg, = ax.plot([], [], 'g-', label='Verde')
    lb, = ax.plot([], [], 'b-', label='Azul')
    ax.set_xlabel('X [cm]'); ax.set_ylabel('Y [cm]')
    ax.set_title('Trayectoria XY de los tres puntos')
    ax.set_aspect('equal','box')
    ax.grid(); ax.legend()
    plt.show(block=False)

    xs = {'red':[], 'green':[], 'blue':[]}
    ys = {'red':[], 'green':[], 'blue':[]}
    prev = {'red':None,'green':None,'blue':None}
    origin = None
    MAX_JUMP = 10.0
    t_start = time.time()

    while True:
        cr, cg, cb, frame = detect()
        t = time.time() - t_start
        if None in (*cr,*cg,*cb):
            cv2.putText(frame,"Tracking lost",(10,30),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
            cv2.imshow("Tracking + Trayectorias", frame)
            if cv2.waitKey(1)&0xFF==ord('q'): break
            continue

        dx1, dy1 = cg[0]-cb[0], cb[1]-cg[1]
        dx2, dy2 = cr[0]-cg[0], cg[1]-cr[1]
        blue_cm  = (0.0, 0.0)
        green_cm = (dx1*s1, dy1*s1)
        red_cm   = (green_cm[0] + dx2*s2, green_cm[1] + dy2*s2)

        if origin is None:
            origin = blue_cm

        rel = {
            'red':   (red_cm[0]-origin[0],   red_cm[1]-origin[1]),
            'green': (green_cm[0]-origin[0], green_cm[1]-origin[1]),
            'blue':  (0.0,                   0.0)
        }

        for color in ('red','green','blue'):
            cur = rel[color]
            p = prev[color]
            if p is None or np.hypot(cur[0]-p[0], cur[1]-p[1])<MAX_JUMP:
                xs[color].append(cur[0]); ys[color].append(cur[1])
                prev[color] = cur
            else:
                xs[color].append(np.nan); ys[color].append(np.nan)

        for pt, col in zip((cr,cg,cb),((0,0,255),(0,255,0),(255,0,0))):
            cv2.circle(frame, pt, 6, col, -1)

        writer.writerow([
            f"{t:.4f}",
            f"{rel['red'][0]:.3f}",   f"{rel['red'][1]:.3f}",
            f"{rel['green'][0]:.3f}", f"{rel['green'][1]:.3f}",
            f"{rel['blue'][0]:.3f}",  f"{rel['blue'][1]:.3f}"
        ])

        # rellenar y suavizar antes de dibujar
        xr, yr = fill_nans(xs['red'], ys['red'])
        xr, yr = smooth(xr), smooth(yr)
        xg, yg = fill_nans(xs['green'], ys['green'])
        xg, yg = smooth(xg), smooth(yg)
        xb, yb = fill_nans(xs['blue'], ys['blue'])
        xb, yb = smooth(xb), smooth(yb)

        lr.set_data(xr, yr)
        lg.set_data(xg, yg)
        lb.set_data(xb, yb)

        # zoom con margen
        ax.relim(); ax.autoscale_view()
        x0,x1 = ax.get_xlim(); y0,y1 = ax.get_ylim()
        ax.set_xlim(x0-2, x1+2); ax.set_ylim(y0-2, y1+2)

        plt.pause(0.001)
        cv2.imshow("Tracking + Trayectorias", frame)
        if cv2.waitKey(1)&0xFF==ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    csv_f.close()
    plt.ioff(); plt.show()

if __name__=="__main__":
    main()
