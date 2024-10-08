Translated GEOMID_61385024_105553168153744 { 
    Translation <1.69529,0.354465,18.0291>
    Geometry Oriented { 
        Primary <0.71694,0.538586,-0.442631>
        Secondary <0.473906,-0.842188,-0.257163>
        Geometry Scaled { 
            Scale <0.6,0.6,0.6>
            Geometry BezierPatch leaf { 
                UDegree 3
                VDegree 3
                CtrlPointMatrix [
                     [ <-0.0153183,0.0439113,-0.0139826,1>, <-0.0851512,0.240738,0.0772178,1>, <-0.131038,0.279686,0.508845,1>, <-0.0379832,0.213685,0.917577,1> ],
                     [ <0.000356436,0.0179964,5.0405e-05,1>, <-0.0419698,0.0633473,0.305836,1>, <-0.0781736,0.0471505,0.688113,1>, <0.0341873,0.08491,1.38917,1> ],
                     [ <-0.000356436,-0.0179964,-5.0405e-05,1>, <-0.0444174,0.0237552,0.317489,1>, <-0.0780897,0.0233952,0.684125,1>, <0.0403175,0.0831102,1.40986,1> ],
                     [ <-0.0196323,-0.0428314,0.00413176,1>, <-0.108456,-0.18592,0.103918,1>, <-0.135263,-0.207506,0.543696,1>, <-0.0547557,-0.0924192,0.923022,1> ]
                ]
                UStride 4
                VStride 4
            }
        }
    }
}


Material green_leaf { 
    Ambient <4,69,4>
    Diffuse 0.768116
    Specular 116
}


Shape SHAPEID_61385024_105553177651520 { 
    Id  61385024
    Geometry  GEOMID_61385024_105553168153744
    Appearance  green_leaf
}


