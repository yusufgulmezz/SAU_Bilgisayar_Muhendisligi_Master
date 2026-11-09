sim = require('sim')

-- ============================================================================
-- BOLUM 0: BASLANGIC AYARLARI VE DONANIM TANIMLAMALARI (INITIALIZATION)
-- Simulasyon basladiginda bir kez calisir. Motor ve sensor adreslerini (handle) alir.
-- ============================================================================
function sysCall_init()
    -- Diferansiyel surus (DDMR) icin sol ve sag tekerlek motorlarini tanimliyoruz
    motorLeft = sim.getObject('/PioneerP3DX/leftMotor')
    motorRight = sim.getObject('/PioneerP3DX/rightMotor')

    -- Robot uzerindeki 16 adet ultrasonik mesafe sensorunu diziye kaydediyoruz
    usensors = {}
    for i = 0, 15 do
        usensors[i+1] = sim.getObject('/PioneerP3DX/visible/ultrasonicSensor['..i..']')
    end
end

-- ============================================================================
-- YARDIMCI FONKSIYON: YAMUK UYELIK FONKSIYONU (TRAPEZOIDAL MEMBERSHIP FUNCTION)
-- Keskin (crisp) sensor mesafesini (x) alir, [a, b, c, d] sinirlarina gore
-- 0 ile 1 arasinda bulanik uyelik derecesine (membership degree) donusturur.
-- ============================================================================
function trapmf(x, a, b, c, d)
    if x <= a or x >= d then return 0 end        -- Kume sinirlari disindaysa uyelik = 0
    if x >= b and x <= c then return 1 end       -- Tam tepe (cekirdek) bolgesindeyse uyelik = 1
    if x < b then return (x - a) / (b - a) end   -- Yukselen kenar: (0'dan 1'e dogrusal artis)
    return (d - x) / (d - c)                     -- Dusen kenar: (1'den 0'a dogrusal azalis)
end

-- ============================================================================
-- ANA DONGU: SENSOR OKUMA VE BULANIK KONTROL ADIMLARI (ACTUATION)
-- Simulasyonun her adiminda (gercek zamanli) tekrar calisir.
-- ============================================================================
function sysCall_actuation()
    if not motorLeft or not motorRight then return end

    -- ------------------------------------------------------------------------
    -- ADIM 1: KESKIN GIRISLERIN OKUNMASI (CRISP INPUTS)
    -- Sensorlerden gelen metre cinsinden veriyi santimetreye (dist * 100) cevirir.
    -- Her bolge icin en yakin engelin mesafesini (minimum degeri) baz alir.
    -- Varsayilan guvenli (engel yok) mesafe 45 cm olarak baslatilir.
    -- ------------------------------------------------------------------------
    local d_l, d_c, d_r = 45.0, 45.0, 45.0

    -- Sol Bolge (Sensor 1, 2, 3): Robotun sol ve sol-caprazini tarar
    for i = 1, 3 do
        local res, dist = sim.readProximitySensor(usensors[i])
        if res > 0 and (dist * 100) < d_l then d_l = dist * 100 end
    end
    
    -- On Bolge (Sensor 4, 5): Robotun tam karsi istikametini tarar
    for i = 4, 5 do
        local res, dist = sim.readProximitySensor(usensors[i])
        if res > 0 and (dist * 100) < d_c then d_c = dist * 100 end
    end
    
    -- Sag Bolge (Sensor 6, 7, 8): Robotun sag ve sag-caprazini tarar
    for i = 6, 8 do
        local res, dist = sim.readProximitySensor(usensors[i])
        if res > 0 and (dist * 100) < d_r then d_r = dist * 100 end
    end

    -- ------------------------------------------------------------------------
    -- ADIM 2: BULANIKLASTIRMA (FUZZIFICATION)
    -- Olculen cm cinsinden mesafeler (d_l, d_c, d_r), dilsel degiskenlere 
    -- ("Yakin" ve "Uzak" bulanik kumelerine) 0.0 - 1.0 araliginda atanir.
    -- ------------------------------------------------------------------------
    -- Sol sensor: 0-15 cm arasi tam Yakin (1), 15-30 cm arasi gecis, 30+ cm tam Uzak (1)
    local l_yakin = trapmf(d_l, 0, 0, 15, 50)
    local l_uzak  = trapmf(d_l, 15, 30, 100, 100)
    
    -- On sensor: Guvenli fren/donus mesafesi icin esik biraz daha erken (20-35 cm) tutulmustur
    local c_yakin = trapmf(d_c, 0, 0, 20, 45)
    local c_uzak  = trapmf(d_c, 20, 35, 100, 100)
    
    -- Sag sensor: 0-15 cm arasi tam Yakin (1), 15-30 cm arasi gecis, 30+ cm tam Uzak (1)
    local r_yakin = trapmf(d_r, 0, 0, 15, 50)
    local r_uzak  = trapmf(d_r, 15, 30, 100, 100)

    -- ------------------------------------------------------------------------
    -- ADIM 3: KURAL TABANI VE BULANIK CIKARIM (FUZZY INFERENCE - T-NORM)
    -- Kurallardaki "VE" (AND) baglaci icin Zadeh'in T-Norm Minimum (math.min)
    -- operatoru kullanilir. Her kural bir ateslenme gucu (k_i) ve 
    -- tekil (singleton) hedef tekerlek hizlari (rad/s) uretir.
    -- ------------------------------------------------------------------------
    
    -- Kural 1: EGER Sol Uzak VE On Uzak VE Sag Uzak ISE -> Tam Ileri Git
    local k1 = math.min(l_uzak, c_uzak, r_uzak)
    local k1_sol_hiz, k1_sag_hiz = 6.0, 6.0

    -- Kural 2: EGER Sol Yakin VE On Yakin VE Sag Uzak ISE -> Saga Keskin Don
    -- (Sol tekerlek ileri iterken sag tekerlek geri cekilerek oldugu yerde kavis verilir)
    local k2 = math.min(l_yakin, c_yakin, r_uzak)
    local k2_sol_hiz, k2_sag_hiz = 3.0, -1.5

    -- Kural 3: EGER Sol Uzak VE On Yakin VE Sag Yakin ISE -> Sola Keskin Don
    local k3 = math.min(l_uzak, c_yakin, r_yakin)
    local k3_sol_hiz, k3_sag_hiz = -1.5, 3.0

    -- Kural 5: EGER Sadece Sol Yakin (On ve Sag Uzak) ISE -> Hafif Saga Yonel
    -- (Duvara paralel ilerlerken sol duvardan uzaklasmak icin yumusak duzeltme)
    local k5 = math.min(l_yakin, c_uzak, r_uzak)
    local k5_sol_hiz, k5_sag_hiz = 3.0, 1.0

    -- Kural 6: EGER Sadece Sag Yakin (Sol ve On Uzak) ISE -> Hafif Sola Yonel
    local k6 = math.min(l_uzak, c_uzak, r_yakin)
    local k6_sol_hiz, k6_sag_hiz = 1.0, 3.0

    -- ------------------------------------------------------------------------
    -- ADIM 4: BERRAKLASTIRMA (DEFUZZIFICATION - AGIRLIKLI ORTALAMA / SUGENO)
    -- Aktiflesen tum kurallarin hedef hizlari, o kuralin ateslenme agirligi (k_i)
    -- ile carpilip toplam agirliga bolunur: V = Toplam(k_i * V_i) / Toplam(k_i)
    -- Bu sayede kural gecislerinde sicrama olmaz, hizlar puruzsuz harmanlanir.
    -- ------------------------------------------------------------------------
    local toplam_agirlik = k1 + k2 + k3 + k5 + k6
    
    local v_l, v_r = 0, 0
    
    if toplam_agirlik > 0 then
        -- Sol ve sag motorlar icin agirlikli ortalama hesabi (Keskin Cikis / Crisp Output)
        v_l = (k1*k1_sol_hiz + k2*k2_sol_hiz + k3*k3_sol_hiz + k5*k5_sol_hiz + k6*k6_sol_hiz) / toplam_agirlik
        v_r = (k1*k1_sag_hiz + k2*k2_sag_hiz + k3*k3_sag_hiz + k5*k5_sag_hiz + k6*k6_sag_hiz) / toplam_agirlik
    else
        -- Guvenlik durumu (Fail-safe): Tanimli kurallar disinda bir durum olusursa robotu durdur
        v_l, v_r = 0, 0
    end

    -- ------------------------------------------------------------------------
    -- ADIM 5: MOTOR SURUCULERINE HIZ KOMUTUNUN GONDERILMESI
    -- ------------------------------------------------------------------------
    sim.setJointTargetVelocity(motorLeft, v_l)
    sim.setJointTargetVelocity(motorRight, v_r)
end