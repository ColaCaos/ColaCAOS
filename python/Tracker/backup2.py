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

def main():
    L1_cm, L2_cm = 23.5, 20.0
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 60)

    csv_f = open('coordenadas_cm.csv', 'w', newline='')
    writer = csv.writer(csv_f)
    writer.writerow([
        't_s',
        'red_x_cm','red_y_cm',
        'green_x_cm','green_y_cm',
        'blue_x_cm','blue_y_cm'
    ])

    cv2.namedWindow("Ajustes", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("TH_RED","Ajustes",47,100,lambda x: None)
    cv2.createTrackbar("TH_GREEN","Ajustes",41,100,lambda x: None)
    cv2.createTrackbar("TH_BLUE","Ajustes",40,100,lambda x: None)
    cv2.createTrackbar("MIN_SUM","Ajustes",71,255,lambda x: None)

    def detect_once():
        ret, frame = cap.read()
        if not ret:
            return None, None, None, None
        blur = cv2.GaussianBlur(frame,(7,7),0).astype(np.float32)
        B,G,R = cv2.split(blur)
        S = B+G+R
        minsum = cv2.getTrackbarPos("MIN_SUM","Ajustes")
        valid = S>minsum
        r_n = np.zeros_like(R); g_n = np.zeros_like(G); b_n = np.zeros_like(B)
        r_n[valid] = R[valid]/S[valid]
        g_n[valid] = G[valid]/S[valid]
        b_n[valid] = B[valid]/S[valid]
        th_r = cv2.getTrackbarPos("TH_RED","Ajustes")/100.0
        th_g = cv2.getTrackbarPos("TH_GREEN","Ajustes")/100.0
        th_b = cv2.getTrackbarPos("TH_BLUE","Ajustes")/100.0
        mask_r = ((r_n>th_r)&(r_n>g_n)&(r_n>b_n)).astype(np.uint8)*255
        mask_g = ((g_n>th_g)&(g_n>r_n)&(g_n>b_n)).astype(np.uint8)*255
        mask_b = ((b_n>th_b)&(b_n>r_n)&(b_n>g_n)).astype(np.uint8)*255
        kern = np.ones((3,3),np.uint8)
        mask_r = cv2.morphologyEx(mask_r,cv2.MORPH_OPEN,kern)
        mask_g = cv2.morphologyEx(mask_g,cv2.MORPH_OPEN,kern)
        mask_b = cv2.morphologyEx(mask_b,cv2.MORPH_OPEN,kern)
        cr = find_centroid(mask_r)
        cg = find_centroid(mask_g)
        cb = find_centroid(mask_b)
        cv2.imshow("Mask Red",mask_r)
        cv2.imshow("Mask Green",mask_g)
        cv2.imshow("Mask Blue",mask_b)
        return cr, cg, cb, frame

    # calibración 5 s
    s1 = s2 = None
    t_cal = time.time()
    while time.time()-t_cal < 5.0:
        cr,cg,cb,_ = detect_once()
        if None in (*cr,*cg,*cb):
            if cv2.waitKey(1)&0xFF==ord('q'):
                cap.release(); return
            continue
        d1 = np.hypot(cg[0]-cb[0],cg[1]-cb[1])
        d2 = np.hypot(cr[0]-cg[0],cr[1]-cg[1])
        if d1>5 and d2>5:
            s1 = L1_cm/d1
            s2 = L2_cm/d2
            print(f"Calibrado: s1={s1:.3f}, s2={s2:.3f}")
            break
    if s1 is None or s2 is None:
        print("Falló calibración"); cap.release(); return

    plt.ion()
    fig,ax = plt.subplots()
    lr,=ax.plot([],[],'r-',label='Rojo')
    lg,=ax.plot([],[],'g-',label='Verde')
    lb,=ax.plot([],[],'b-',label='Azul')
    ax.set_xlabel('X [cm]'); ax.set_ylabel('Y [cm]')
    ax.set_title('Trayectorias en cm')
    ax.legend(); ax.grid(); ax.invert_yaxis()
    plt.show(block=False)

    alpha = 0.3
    prev_r = prev_g = (0.0, 0.0)

    red_xs, red_ys     = [],[]
    green_xs, green_ys = [],[]
    blue_xs, blue_ys   = [],[]

    t0 = time.time()
    while True:
        cr,cg,cb,frame = detect_once()
        t = time.time()-t0
        if None in (*cr,*cg,*cb):
            cv2.putText(frame,"Tracking lost",(10,30),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
            cv2.imshow("Tracking + Trayectorias",frame)
            if cv2.waitKey(1)&0xFF==ord('q'): break
            continue

        bx,by = cb
        gx,gy = cg
        rx,ry = cr
        dx1,dy1 = gx-bx, gy-by
        dx2,dy2 = rx-gx, ry-gy
        blue_cm  = (0.0,0.0)
        green_raw = (dx1*s1, dy1*s1)
        red_raw   = (green_raw[0]+dx2*s2,
                     green_raw[1]+dy2*s2)

        # EWMA smoothing
        green_sm = (alpha*green_raw[0] + (1-alpha)*prev_g[0],
                    alpha*green_raw[1] + (1-alpha)*prev_g[1])
        red_sm   = (alpha*red_raw[0]   + (1-alpha)*prev_r[0],
                    alpha*red_raw[1]   + (1-alpha)*prev_r[1])

        prev_g, prev_r = green_sm, red_sm

        green_cm = green_sm
        red_cm   = red_sm

        for (x,y),col in zip((cr,cg,cb),
                            ((0,0,255),(0,255,0),(255,0,0))):
            cv2.circle(frame,(x,y),6,col,-1)

        writer.writerow([
            f"{t:.4f}",
            f"{red_cm[0]:.3f}",f"{red_cm[1]:.3f}",
            f"{green_cm[0]:.3f}",f"{green_cm[1]:.3f}",
            f"{blue_cm[0]:.3f}",f"{blue_cm[1]:.3f}"
        ])
        csv_f.flush()

        red_xs.append(red_cm[0]); red_ys.append(red_cm[1])
        green_xs.append(green_cm[0]); green_ys.append(green_cm[1])
        blue_xs.append(blue_cm[0]); blue_ys.append(blue_cm[1])

        lr.set_data(red_xs, red_ys)
        lg.set_data(green_xs, green_ys)
        lb.set_data(blue_xs, blue_ys)
        ax.relim(); ax.autoscale_view()
        plt.pause(0.001)

        cv2.imshow("Tracking + Trayectorias",frame)
        if cv2.waitKey(1)&0xFF==ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    csv_f.close()
    plt.ioff()
    plt.show()

if __name__=="__main__":
    main()
