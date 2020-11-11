#!/bin/bash

if [ $1 == 'sanity_check' ]
then
    python sanity_check.py prepare_data
    python sanity_check.py AdaIN
    python sanity_check.py Projection
    python sanity_check.py mean_std_2d
    python sanity_check.py BLClassifier
    python sanity_check.py ResBlock
    python sanity_check.py HoloDiscriminator
    python sanity_check.py RigidTransform3d
    python sanity_check.py HoloGenerator
    python sanity_check.py mlp
    if ! [ -z $2 ]
    then
        python sanity_check.py train_discriminator --gpu
    fi
fi