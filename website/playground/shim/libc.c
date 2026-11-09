/*
 * Tiny freestanding C library for the WebAssembly build of the TIL VM:
 * the <string.h> functions (see string.h). Math lives in math.h (WebAssembly
 * instructions and JavaScript Math imports). No allocation, no I/O.
 *
 * memcpy/memmove/memset use __builtin_* calls, which clang lowers to the
 * WebAssembly bulk-memory instructions (build.sh passes -mbulk-memory).
 */
#include <string.h>

void *memcpy(void *dst, const void *src, size_t n) {
    return __builtin_memcpy(dst, src, n);
}

void *memmove(void *dst, const void *src, size_t n) {
    return __builtin_memmove(dst, src, n);
}

void *memset(void *dst, int c, size_t n) {
    return __builtin_memset(dst, c, n);
}

int memcmp(const void *a, const void *b, size_t n) {
    const unsigned char *p = (const unsigned char *)a, *q = (const unsigned char *)b;
    size_t i;
    for (i = 0; i < n; i++)
        if (p[i] != q[i]) return p[i] < q[i] ? -1 : 1;
    return 0;
}

void *memchr(const void *s, int c, size_t n) {
    const unsigned char *p = (const unsigned char *)s;
    size_t i;
    for (i = 0; i < n; i++)
        if (p[i] == (unsigned char)c) return (void *)(p + i);
    return 0;
}

size_t strlen(const char *s) {
    size_t n = 0;
    while (s[n]) n++;
    return n;
}

size_t strnlen(const char *s, size_t max) {
    size_t n = 0;
    while (n < max && s[n]) n++;
    return n;
}

char *strcpy(char *dst, const char *src) {
    size_t i = 0;
    while ((dst[i] = src[i]) != 0) i++;
    return dst;
}

char *strncpy(char *dst, const char *src, size_t n) {
    size_t i = 0;
    for (; i < n && src[i]; i++) dst[i] = src[i];
    for (; i < n; i++) dst[i] = 0;
    return dst;
}

char *strcat(char *dst, const char *src) {
    strcpy(dst + strlen(dst), src);
    return dst;
}

char *strncat(char *dst, const char *src, size_t n) {
    size_t d = strlen(dst), i = 0;
    for (; i < n && src[i]; i++) dst[d + i] = src[i];
    dst[d + i] = 0;
    return dst;
}

int strcmp(const char *a, const char *b) {
    const unsigned char *p = (const unsigned char *)a, *q = (const unsigned char *)b;
    while (*p && *p == *q) { p++; q++; }
    return *p == *q ? 0 : (*p < *q ? -1 : 1);
}

int strncmp(const char *a, const char *b, size_t n) {
    const unsigned char *p = (const unsigned char *)a, *q = (const unsigned char *)b;
    size_t i;
    for (i = 0; i < n; i++) {
        if (p[i] != q[i]) return p[i] < q[i] ? -1 : 1;
        if (!p[i]) return 0;
    }
    return 0;
}

char *strchr(const char *s, int c) {
    for (;; s++) {
        if (*s == (char)c) return (char *)s;
        if (!*s) return 0;
    }
}

char *strrchr(const char *s, int c) {
    const char *last = 0;
    for (;; s++) {
        if (*s == (char)c) last = s;
        if (!*s) return (char *)last;
    }
}

char *strstr(const char *haystack, const char *needle) {
    size_t n = strlen(needle);
    if (!n) return (char *)haystack;
    for (; *haystack; haystack++)
        if (*haystack == *needle && strncmp(haystack, needle, n) == 0) return (char *)haystack;
    return 0;
}
