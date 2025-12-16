/* dgs_global.h
 *
 * contains various constants/structs/definitions that are used globally
 */

#ifndef DGS_GLOBAL_H
#define DGS_GLOBAL_H

#include <stdint.h>
#include <string.h>

//-------------------------------------------//

#ifdef __cplusplus
	#define DGS_NOMANGLE extern "C"
#else
	#define DGS_NOMANGLE
#endif

#ifdef _WIN32
	#define DGS_API DGS_NOMANGLE __declspec(dllexport)
#else
	#define DGS_API DGS_NOMANGLE __attribute__((visibility("default")))
#endif

//-------------------------------------------//

#if !defined(DGS) || !defined(DGS_FREE) || !defined(QOBJ_FREE)
	#include <stdlib.h>

	#define DGS_MALLOC(s) malloc(s)
	#define DGS_FREE(p) free(p)
	#define DGS_REALLOC(p, s) realloc(p, s)
#endif

#define DGS_STRUCTURE_CLEAR(s) memset(s, 0, sizeof(*s))

//-------------------------------------------//

typedef uint8_t dgs_bool_t;

#define DGS_TRUE 1
#define DGS_FALSE 0

//-------------------------------------------//

#define DGS_MAX(a,b) (((a) > (b)) ? (a) : (b))
#define DGS_MIN(a,b) (((a) < (b)) ? (a) : (b))
#define DGS_ABS(a) ((a) > 0 ? (a) : -(a))
#define DGS_CLAMP(v,a,b) DGS_MIN(DGS_MAX(v, a), b)

#define DGS_PI 3.14159265358979323846f
#define DGS_EPSILON 0.0001f

#endif //#ifndef DGS_GLOBAL_H
