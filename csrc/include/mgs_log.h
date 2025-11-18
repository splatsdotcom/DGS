/* mgs_log.h
 * 
 * contains functions/macros for logging
 */

#ifndef MGS_LOG_H
#define MGS_LOG_H

#include <stdio.h>
#include "mgs_error.h"

//-------------------------------------------//

#ifndef MGS_LOG_ERROR
	#define MGS_LOG_ERROR(msg) printf("MGS ERROR: \"%s\" in %s at line %i\n", msg, __FILENAME__, __LINE__)
#endif
#ifndef MGS_LOG_WARNING
	#define MGS_LOG_WARNING(msg) printf("MGS WARNING: \"%s\" in %s at line %i\n", msg, __FILENAME__, __LINE__)
#endif
#ifndef MGS_LOG_STACK_POSITION
	#define MGS_LOG_STACK_POSITION() printf("\tin %s at line %i\n", __FILENAME__, __LINE__)
#endif

#ifdef MGS_DEBUG
	#define MGS_ASSERT(c,m) if(!(c)) { printf("MGS ASSERTION FAIL: %s\n", m); exit(-1); }
#else
	#define MGS_ASSERT(c,m)
#endif
#define MGS_STATIC_ASSERT(condition, message) typedef char mgs_static_assertion_##message[(condition) ? 1 : -1]

#define MGS_MALLOC_CHECK(p) {                                                   \
                             	if((p) == NULL)                                 \
                             	{                                               \
                             		MGS_LOG_ERROR("failed to allocate memory"); \
                             		retval = MGS_ERROR_OUT_OF_MEMORY;           \
                             		goto cleanup;                               \
                             	}                                               \
                             }

#define MGS_ERROR_PROPAGATE(s) {                                  \
                                	retval = s;                   \
                                	if(retval != MGS_SUCCESS)     \
                                	{                             \
                                		MGS_LOG_STACK_POSITION(); \
                                		goto cleanup;             \
                                	}                             \
                                }

#endif //#ifndef MGS_LOG_H