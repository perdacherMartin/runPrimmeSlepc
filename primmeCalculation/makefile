#
#  makefile
#
#  Created by Martin Perdacher on 2013-09-21.
#  Copyright (c) 2013 . All rights reserved.
#

driver: driver.o chkopts
	-${CLINKER} -o driver driver.o ${SLEPC_LIB} -g 
	${RM} driver.o

include ${SLEPC_DIR}/conf/slepc_common
