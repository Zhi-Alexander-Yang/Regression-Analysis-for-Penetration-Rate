# Select variables of interest
variables_of_interest = ["Penetration rate", "Percussion pressure", "Feed pressure", "Flush air pressure", "Rotation pressure", "Dampening pressure"]

# Calculate correlation
correlation_matrix = df[variables_of_interest].corr()

# Display correlation matrix
correlation_matrix


#RESULT
#                     Penetration rate  Percussion pressure  Feed pressure  \
#Penetration rate             1.000000            -0.205730      -0.287867   
#Percussion pressure         -0.205730             1.000000       0.946751   
#Feed pressure               -0.287867             0.946751       1.000000   
#Flush air pressure          -0.061479             0.339813       0.333079   
#Rotation pressure           -0.048222             0.424689       0.459603   
#Dampening pressure          -0.244864             0.877682       0.944171   
#
#                     Flush air pressure  Rotation pressure  Dampening pressure  
#Penetration rate              -0.061479          -0.048222           -0.244864  
#Percussion pressure            0.339813           0.424689            0.877682  
#Feed pressure                  0.333079           0.459603            0.944171  
#Flush air pressure             1.000000           0.515543            0.237512  
#Rotation pressure              0.515543           1.000000            0.350799  
#Dampening pressure             0.237512           0.350799            1.0000