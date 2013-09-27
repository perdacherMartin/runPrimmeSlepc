#!/bin/bash

EXECUTIONFILE=callLapackSub.sh
PREFIX=CaFe2As2.small
OUTPUTDIR=$PREFIX.primmein2
LAPACKEVEC=lapack
SLAPCEVEC=slepc
STDDIR=std
RELERRORDIR=relError
TIMINGDIR=timing
HMATDIR=HMat
HMATPREFIX=hmat
SMATDIR=SMat
SMATPREFIX=smat
LOGDIR=logs
N=1912
NEV=136

mkdir -p $OUTPUTDIR
mkdir -p $OUTPUTDIR/$STDDIR
mkdir -p $OUTPUTDIR/$LAPACKEVEC
mkdir -p $OUTPUTDIR/$SLAPCEVEC
mkdir -p $OUTPUTDIR/$RELERRORDIR
mkdir -p $OUTPUTDIR/$TIMINGDIR
mkdir -p $OUTPUTDIR/$LOGDIR

for i in {1..30}
do
	export EXECUTIONFILE=callLapackSub$i.sh
	NUMBER=$(printf '%3d%3d' 1 $i)
	LEADINGZERO=$(printf '%03d%03d' 1 $i)
	
	echo "#!/bin/bash" > $EXECUTIONFILE
	echo "#MSUB -l nodes=1:ppn=8" >> $EXECUTIONFILE
	echo "#MSUB -l walltime=4:0:00" >> $EXECUTIONFILE
	echo "#MSUB -e $OUTPUTDIR/$LOGDIR/callLapack.err.$i.txt" >> $EXECUTIONFILE
	echo "#MSUB -o $OUTPUTDIR/$LOGDIR/callLapack.out.$i.txt" >> $EXECUTIONFILE
	echo "" >> $EXECUTIONFILE
	echo "SAVEIFS=\$IFS" >> $EXECUTIONFILE
	echo "IFS=\$(echo -en \"\\n\")" >> $EXECUTIONFILE	
	
	HMAT=$(echo "${HMATDIR}/${HMATPREFIX}${NUMBER}" | sed 's/\ /\\\ /g')
	SMAT=$(echo "${SMATDIR}/${SMATPREFIX}${NUMBER}" | sed 's/\ /\\\ /g')
	echo "" >> $EXECUTIONFILE
	echo "export MKL_NUM_THREADS=8" >> $EXECUTIONFILE
	echo -n "mpiexec -np 1 lapack -hmat $HMAT -smat $SMAT -n $N -nev $NEV -stw $OUTPUTDIR/$STDDIR/${PREFIX}.standard.${LEADINGZERO}.dat " >> $EXECUTIONFILE	
	echo -n " -evecw $OUTPUTDIR/$LAPACKEVEC/$PREFIX.lapack.${LEADINGZERO}.dat -relLapack $OUTPUTDIR/$RELERRORDIR/$PREFIX.lapack.err.${LEADINGZERO}.dat " >> $EXECUTIONFILE	
	#echo -n " -evecSlepcw $OUTPUTDIR/$SLAPCEVEC/$PREFIX.slepc.${LEADINGZERO}.dat -relSlepc $OUTPUTDIR/$RELERRORDIR/$PREFIX.slepc.error${LEADINGZERO}.csv " >> $EXECUTIONFILE 
	echo " -timing $OUTPUTDIR/$TIMINGDIR/$PREFIX.time.${LEADINGZERO}.csv -evalFile $OUTPUTDIR/${PREFIX}.${LEADINGZERO}.evalues.lapack.dat" >> $EXECUTIONFILE	
	echo "" >> $EXECUTIONFILE
	echo "\$IFS=\$SAVEIFS" >> $EXECUTIONFILE
	
	chmod 755 $EXECUTIONFILE
	msub $EXECUTIONFILE
	sleep 15
done


