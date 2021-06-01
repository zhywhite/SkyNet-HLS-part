#include "net.h"

FIX_FM batch_norm(int bn_type,int data)
{
	FIX_FM result;
    if (bn_type == 1)  result = relu6_single(data);
	return result;
} 

FIX_FM relu6_single( FIX_FM d ) {
	if( d > 6 )
		return 6;
	if( d < 0 )
		return 0;
	return d;
}

