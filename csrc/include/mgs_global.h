/* mgs_global.h
 *
 * contains various constants/structs/definitions that are used globally
 */

#ifndef MGS_GLOBAL_H
#define MGS_GLOBAL_H

#include <stdint.h>
#include <string.h>

//-------------------------------------------//

#ifdef __cplusplus
	#define MGS_NOMANGLE extern "C"
#else
	#define MGS_NOMANGLE
#endif

#ifdef _WIN32
	#define MGS_API MGS_NOMANGLE __declspec(dllexport)
#else
	#define MGS_API MGS_NOMANGLE __attribute__((visibility("default")))
#endif

//-------------------------------------------//

#if !defined(MGS) || !defined(MGS_FREE) || !defined(QOBJ_FREE)
	#include <stdlib.h>

	#define MGS_MALLOC(s) malloc(s)
	#define MGS_FREE(p) free(p)
	#define MGS_REALLOC(p, s) realloc(p, s)
#endif

#define MGS_STRUCTURE_CLEAR(s) memset(s, 0, sizeof(*s))

//-------------------------------------------//

typedef uint8_t mgs_bool_t;

#define MGS_TRUE 1
#define MGS_FALSE 0

//-------------------------------------------//

#define MGS_MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MGS_MIN(a,b) (((a) < (b)) ? (a) : (b))
#define MGS_ABS(a) ((a) > 0 ? (a) : -(a))
#define MGS_CLAMP(v,a,b) MGS_MIN(MGS_MAX(v, a), b)

#define MGS_PI 3.14159265358979323846f
#define MGS_EPSILON 0.0001f

#endif //#ifndef MGS_GLOBAL_H
