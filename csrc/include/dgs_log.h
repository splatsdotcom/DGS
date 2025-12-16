/* dgs_log.h
 * 
 * contains functions/macros for logging
 */

#ifndef DGS_LOG_H
#define DGS_LOG_H

#include <stdio.h>
#include "dgs_error.h"

//-------------------------------------------//

#ifndef DGS_LOG_ERROR
	#define DGS_LOG_ERROR(msg) printf("DGS ERROR: \"%s\" in %s at line %i\n", msg, __FILENAME__, __LINE__)
#endif
#ifndef DGS_LOG_WARNING
	#define DGS_LOG_WARNING(msg) printf("DGS WARNING: \"%s\" in %s at line %i\n", msg, __FILENAME__, __LINE__)
#endif
#ifndef DGS_LOG_STACK_POSITION
	#define DGS_LOG_STACK_POSITION() printf("\tin %s at line %i\n", __FILENAME__, __LINE__)
#endif

#ifdef DGS_DEBUG
	#define DGS_ASSERT(c,m) if(!(c)) { printf("DGS ASSERTION FAIL: %s\n", m); exit(-1); }
#else
	#define DGS_ASSERT(c,m)
#endif
#define DGS_STATIC_ASSERT(condition, message) typedef char dgs_static_assertion_##message[(condition) ? 1 : -1]

#define DGS_MALLOC_CHECK(p) {                                                   \
                             	if((p) == NULL)                                 \
                             	{                                               \
                             		DGS_LOG_ERROR("failed to allocate memory"); \
                             		retval = DGS_ERROR_OUT_OF_MEMORY;           \
                             		goto cleanup;                               \
                             	}                                               \
                             }

#define DGS_ERROR_PROPAGATE(s) {                                  \
                                	retval = s;                   \
                                	if(retval != DGS_SUCCESS)     \
                                	{                             \
                                		DGS_LOG_STACK_POSITION(); \
                                		goto cleanup;             \
                                	}                             \
                                }

#endif //#ifndef DGS_LOG_H