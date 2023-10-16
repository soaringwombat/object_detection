import cv2

# カメラを開く
cap = cv2.VideoCapture(0)

# 背景差分器を作成する
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    # フレームを読み込む
    ret, frame = cap.read()

    if not ret:
        break

    # 背景差分を計算する
    fgmask = fgbg.apply(frame)

    # 輪郭を検出する
    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 輪郭の中心座標を格納するリストを作成する
    centers = []

    # 輪郭を描画する
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # 輪郭の中心座標を計算する
            center_x = x + w // 2
            center_y = y + h // 2
            # 中心座標をリストに追加する
            centers.append((center_x, center_y))

    # 中心座標の平均を計算する
    if len(centers) > 0:
        avg_x = sum([center[0] for center in centers]) // len(centers)
        avg_y = sum([center[1] for center in centers]) // len(centers)
        # 平均座標を表示する
        print("Average coordinates: ({}, {})".format(avg_x, avg_y))

    # 結果を表示する
    cv2.imshow('frame', frame)
    cv2.imshow('fgmask', fgmask)

    # 終了するためのキー入力を待つ
    if cv2.waitKey(30) & 0xff == ord('q'):
        break

# 後処理
cap.release()
cv2.destroyAllWindows()
