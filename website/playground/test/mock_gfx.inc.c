/*
 * TEST ONLY - minimal screen/key natives for testing the playground's host
 * hooks (canvas, keys, wait_key pauses) while vm/tilvm.c does not implement
 * natives 40-53 yet. tests/test_playground.py pastes this into a temporary
 * copy of vm/tilvm.c (never into the real file) and only when that file has
 * no `case N_SCREEN_INIT:`. Once the real graphics natives are merged this
 * file is unused. Shapes are simplified (text is drawn as 5x7 blocks).
 */
static long mock_w, mock_h;

static void mock_px(tilvm *vm, long x, long y, tv_int c) {
    if (x >= 0 && y >= 0 && x < mock_w && y < mock_h)
        vm->host.gfx_pixel(vm->ctx, (int)x, (int)y, (unsigned long)(c & 0xFFFFFF));
}

static void mock_fill(tilvm *vm, long x0, long y0, long x1, long y1, tv_int c) {
    long x, y;
    for (y = y0; y < y1; y++)
        for (x = x0; x < x1; x++) mock_px(vm, x, y, c);
}

static int mock_gfx(tilvm *vm, uint8_t id, uint16_t base, uint8_t argc, tv_val *res) {
    tv_val *a = &STK[base];
    uint8_t k;
    for (k = 0; k < argc; k++)
        if (a[k].t != TV_INT && !(id == N_DRAW_TEXT && k == 2)) return fail(vm, "mock graphics: int arguments expected");
    if (id == N_KEY_PRESSED || id == N_WAIT_KEY) {
        if (!vm->host.key_pressed || !vm->host.wait_key) return fail(vm, "keys are not available on this device");
        set_int(res, (tv_int)(id == N_WAIT_KEY ? vm->host.wait_key(vm->ctx) : vm->host.key_pressed(vm->ctx)));
        return 1;
    }
    if (!vm->host.gfx_pixel || !vm->host.gfx_init) return fail(vm, "the screen is not available on this device");
    if (id == N_SCREEN_WIDTH) { set_int(res, (tv_int)mock_w); return 1; }
    if (id == N_SCREEN_HEIGHT) { set_int(res, (tv_int)mock_h); return 1; }
    if (id == N_SCREEN_INIT) {
        if (a[0].u.i < 1 || a[1].u.i < 1 || a[0].u.i > 4096 || a[1].u.i > 4096) return fail(vm, "bad screen size");
        mock_w = (long)a[0].u.i;
        mock_h = (long)a[1].u.i;
        vm->host.gfx_init(vm->ctx, (int)mock_w, (int)mock_h);
        return 1;
    }
    if (!mock_w) return fail(vm, "screen not initialized (call screen_init)");
    switch (id) {
    case N_SCREEN_CLEAR: mock_fill(vm, 0, 0, mock_w, mock_h, argc ? a[0].u.i : 0); break;
    case N_DRAW_PIXEL: mock_px(vm, (long)a[0].u.i, (long)a[1].u.i, a[2].u.i); break;
    case N_FILL_RECT:
        mock_fill(vm, (long)a[0].u.i, (long)a[1].u.i, (long)(a[0].u.i + a[2].u.i), (long)(a[1].u.i + a[3].u.i), a[4].u.i);
        break;
    case N_DRAW_RECT: {
        long x = (long)a[0].u.i, y = (long)a[1].u.i, w = (long)a[2].u.i, h = (long)a[3].u.i;
        mock_fill(vm, x, y, x + w, y + 1, a[4].u.i);
        mock_fill(vm, x, y + h - 1, x + w, y + h, a[4].u.i);
        mock_fill(vm, x, y, x + 1, y + h, a[4].u.i);
        mock_fill(vm, x + w - 1, y, x + w, y + h, a[4].u.i);
        break;
    }
    case N_DRAW_LINE: {
        long x0 = (long)a[0].u.i, y0 = (long)a[1].u.i, x1 = (long)a[2].u.i, y1 = (long)a[3].u.i;
        long dx = x1 > x0 ? x1 - x0 : x0 - x1, dy = y1 > y0 ? y0 - y1 : y1 - y0;
        long sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1, err = dx + dy;
        for (;;) {
            long e2 = 2 * err;
            mock_px(vm, x0, y0, a[4].u.i);
            if (x0 == x1 && y0 == y1) break;
            if (e2 >= dy) { err += dy; x0 += sx; }
            if (e2 <= dx) { err += dx; y0 += sy; }
        }
        break;
    }
    case N_DRAW_CIRCLE: {
        long cx = (long)a[0].u.i, cy = (long)a[1].u.i, r = (long)a[2].u.i, x = r, y = 0, e = 1 - r;
        while (x >= y) {
            mock_px(vm, cx + x, cy + y, a[3].u.i); mock_px(vm, cx - x, cy + y, a[3].u.i);
            mock_px(vm, cx + x, cy - y, a[3].u.i); mock_px(vm, cx - x, cy - y, a[3].u.i);
            mock_px(vm, cx + y, cy + x, a[3].u.i); mock_px(vm, cx - y, cy + x, a[3].u.i);
            mock_px(vm, cx + y, cy - x, a[3].u.i); mock_px(vm, cx - y, cy - x, a[3].u.i);
            y++;
            if (e < 0) e += 2 * y + 1; else { x--; e += 2 * (y - x) + 1; }
        }
        break;
    }
    case N_DRAW_TEXT: {
        uint32_t n, i;
        const char *s;
        if (!to_str_slot(vm, (uint16_t)(base + 2))) return 0;
        s = str_data(vm, &STK[base + 2], &n);
        for (i = 0; i < n; i++)
            if (s[i] != ' ')
                mock_fill(vm, (long)a[0].u.i + 6 * (long)i, (long)a[1].u.i, (long)a[0].u.i + 6 * (long)i + 5,
                          (long)a[1].u.i + 7, a[3].u.i);
        break;
    }
    case N_GET_PIXEL:
        if (a[0].u.i >= 0 && a[1].u.i >= 0 && a[0].u.i < mock_w && a[1].u.i < mock_h && vm->host.gfx_get)
            set_int(res, (tv_int)vm->host.gfx_get(vm->ctx, (int)a[0].u.i, (int)a[1].u.i));
        else
            set_int(res, 0);
        break;
    case N_SCREEN_SHOW:
        if (vm->host.gfx_show) vm->host.gfx_show(vm->ctx);
        break;
    default:
        break;
    }
    return 1;
}
