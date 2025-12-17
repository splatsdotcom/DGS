/* dgs_format.h
 *
 * contains definitions/constants related to the .dgs file format
 */

#ifndef DGS_FORMAT_H
#define DGS_FORMAT_H

#include <stdint.h>

//-------------------------------------------//

#define DGS_MAKE_VERSION(major, minor, patch) (((major) << 22) | ((minor) << 12) | (patch))

#define DGS_MAGIC_WORD (('s' << 24) | ('p' << 16) | ('l' << 8) | ('g'))
#define DGS_VERSION (DGS_MAKE_VERSION(1, 0, 1))

//-------------------------------------------//

typedef struct DGSfileHeader
{
	uint32_t magicWord;
	uint32_t version;
} DGSfileHeader;

typedef struct DGSmetadata
{
	float duration;
} DGSmetadata;

#endif // #ifndef DGS_FORMAT_H