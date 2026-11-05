/*
 * Freestanding <math.h> for the WebAssembly build of the TIL VM.
 *
 * sqrt/floor/ceil/trunc/fabs are single WebAssembly instructions; fmod is the
 * JavaScript `%` operator (exact, same sign rule as C); the transcendental
 * functions are imported from the browser's Math object (Math.pow, Math.sin,
 * ...). Their special cases (inf, nan, zeros) follow C99 Annex F - pow() fixes
 * the two where JavaScript differs - but the last bit of ordinary results
 * often differs from the desktop C library (see website/playground/README.md).
 */
#ifndef TILWASM_MATH_H
#define TILWASM_MATH_H

#define TILWASM_JS_MATH(name) __attribute__((import_module("Math"), import_name(name)))
#define TILWASM_JS_ENV(name) __attribute__((import_module("env"), import_name(name)))

#define HUGE_VAL (__builtin_huge_val())
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
#define isnan(x) __builtin_isnan(x)
#define isinf(x) __builtin_isinf(x)
#define isfinite(x) __builtin_isfinite(x)
#define signbit(x) __builtin_signbit(x)

/* WebAssembly instructions */
static inline double sqrt(double x) { return __builtin_sqrt(x); }
static inline double floor(double x) { return __builtin_floor(x); }
static inline double ceil(double x) { return __builtin_ceil(x); }
static inline double trunc(double x) { return __builtin_trunc(x); }
static inline double fabs(double x) { return __builtin_fabs(x); }
static inline double rint(double x) { return __builtin_rint(x); }
static inline double nearbyint(double x) { return __builtin_nearbyint(x); }

/* halfway cases away from zero, like C99 round() */
static inline double round(double x) {
    double t = __builtin_trunc(x);
    if (__builtin_fabs(x - t) >= 0.5) t += x < 0 ? -1.0 : 1.0;
    return t;
}

/* NaN arguments are ignored, like C99 fmin/fmax */
static inline double fmin(double x, double y) { return (x < y || y != y) ? x : y; }
static inline double fmax(double x, double y) { return (x > y || y != y) ? x : y; }

/* JavaScript: x % y */
double fmod(double x, double y) TILWASM_JS_ENV("fmod");

/* JavaScript Math.* */
double tilwasm_math_pow(double x, double y) TILWASM_JS_MATH("pow");

/* C99 (Annex F) and the reference VM: pow(+1, y) is 1 for every y, NaN too,
 * and pow(-1, +-inf) is 1; JavaScript's Math.pow returns NaN for these. */
static inline double pow(double x, double y) {
    if (x == 1.0 || (x == -1.0 && __builtin_isinf(y))) return 1.0;
    return tilwasm_math_pow(x, y);
}
double exp(double x) TILWASM_JS_MATH("exp");
double expm1(double x) TILWASM_JS_MATH("expm1");
double log(double x) TILWASM_JS_MATH("log");
double log10(double x) TILWASM_JS_MATH("log10");
double log2(double x) TILWASM_JS_MATH("log2");
double log1p(double x) TILWASM_JS_MATH("log1p");
double sin(double x) TILWASM_JS_MATH("sin");
double cos(double x) TILWASM_JS_MATH("cos");
double tan(double x) TILWASM_JS_MATH("tan");
double asin(double x) TILWASM_JS_MATH("asin");
double acos(double x) TILWASM_JS_MATH("acos");
double atan(double x) TILWASM_JS_MATH("atan");
double atan2(double y, double x) TILWASM_JS_MATH("atan2");
double sinh(double x) TILWASM_JS_MATH("sinh");
double cosh(double x) TILWASM_JS_MATH("cosh");
double tanh(double x) TILWASM_JS_MATH("tanh");
double asinh(double x) TILWASM_JS_MATH("asinh");
double acosh(double x) TILWASM_JS_MATH("acosh");
double atanh(double x) TILWASM_JS_MATH("atanh");
double cbrt(double x) TILWASM_JS_MATH("cbrt");
double hypot(double x, double y) TILWASM_JS_MATH("hypot");

#endif
