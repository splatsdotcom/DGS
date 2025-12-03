/* mgs_format.h
 *
 * contains definitions/constants related to the .mgs file format
 */

#ifndef MGS_FORMAT_H
#define MGS_FORMAT_H

#include <stdint.h>

//-------------------------------------------//

#define MGS_MAKE_VERSION(major, minor, patch) (((major) << 22) | ((minor) << 12) | (patch))

#define MGS_MAGIC_WORD (('s' << 24) | ('p' << 16) | ('l' << 8) | ('g'))
#define MGS_VERSION (MGS_MAKE_VERSION(0, 0, 1))

//-------------------------------------------//

typedef struct MGSfileHeader
{
	uint32_t magicWord;
	uint32_t version;
} MGSfileHeader;

typedef struct MGSmetadata
{
	float duration;
} MGSmetadata;

#endif // #ifndef MGS_FORMAT_H