/* 
 * $smu-mark$ 
 * $name: sendhcmp.c$ 
 * $author: Salvatore Sanfilippo <antirez@invece.org>$ 
 * $copyright: Copyright (C) 1999 by Salvatore Sanfilippo$ 
 * $license: This software is under GPL version 2 of license$ 
 * $date: Fri Nov  5 11:55:49 MET 1999$ 
 * $rev: 4$ 
 */
 
/*
 * Revised for Windows: Rob Turpin <rgturpin@epop3.com> 
 *                      8/22/2004          
 */ 

#include <string.h>
#include <assert.h>
#include <signal.h> /* SIGALARM macro */

#ifndef WIN32
#include <unistd.h>
#else
#include <io.h>
#endif

#include "hping2.h"
#include "globals.h"

#define MUST_BE_UNREACHED 0

void send_hcmp(__u8 type, __u32 arg)
{
	static struct hcmphdr hcmph; /* static because we export this */
				     /* to data_handler() */

	data_size = signlen + sizeof(struct hcmphdr);

	/* build hcmp header */
	memset(&hcmph, 0, sizeof(hcmph));
	hcmph.type = type;
	switch (type) {
	case HCMP_RESTART:
		hcmph.typedep.seqnum = htons((__u16) arg);
		break;
	case HCMP_SOURCE_QUENCH:
	case HCMP_SOURCE_STIRUP:
		hcmph.typedep.usec = htonl(arg);
		break;
	default:
		assert(MUST_BE_UNREACHED);
	}

	/* use hcmphdr_p to transmit hcmph to data_handler() */
	hcmphdr_p = &hcmph;
#ifndef WIN32
	kill(getpid(), SIGALRM); /* send hcmp */
#else
  send_packet(0, 0, 0, 0, 0);
#endif

  return;
}
