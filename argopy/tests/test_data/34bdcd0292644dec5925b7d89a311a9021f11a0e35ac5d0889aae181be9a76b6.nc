CDF       
      	DATE_TIME         	STRING256         STRING64   @   STRING32       STRING16      STRING8       STRING4       STRING2       N_PROF        N_PARAM       N_LEVELS   C   N_CALIB       	N_HISTORY                title         Argo float vertical profile    institution       FR GDAC    source        
Argo float     history       2022-06-01T23:47:07Z creation      
references        (http://www.argodatamgt.org/Documentation   user_manual_version       3.1    Conventions       Argo-3.1 CF-1.6    featureType       trajectoryProfile         @   	DATA_TYPE                  	long_name         	Data type      conventions       Argo reference table 1     
_FillValue                    6x   FORMAT_VERSION                 	long_name         File format version    
_FillValue                    6�   HANDBOOK_VERSION               	long_name         Data handbook version      
_FillValue                    6�   REFERENCE_DATE_TIME                 	long_name         !Date of reference for Julian days      conventions       YYYYMMDDHHMISS     
_FillValue                    6�   DATE_CREATION                   	long_name         Date of file creation      conventions       YYYYMMDDHHMISS     
_FillValue                    6�   DATE_UPDATE                 	long_name         Date of update of this file    conventions       YYYYMMDDHHMISS     
_FillValue                    6�   PLATFORM_NUMBER                   	long_name         Float unique identifier    conventions       WMO float identifier : A9IIIII     
_FillValue                  �  6�   PROJECT_NAME                  	long_name         Name of the project    
_FillValue                 �  7x   PI_NAME                   	long_name         "Name of the principal investigator     
_FillValue                 �  =8   STATION_PARAMETERS           	            	long_name         ,List of available parameters for the station   conventions       Argo reference table 3     
_FillValue                 P  B�   CYCLE_NUMBER               	long_name         Float cycle number     conventions       =0...N, 0 : launch cycle (if exists), 1 : first complete cycle      
_FillValue         ��      \  GH   	DIRECTION                  	long_name         !Direction of the station profiles      conventions       -A: ascending profiles, D: descending profiles      
_FillValue                    G�   DATA_CENTRE                   	long_name         .Data centre in charge of float data processing     conventions       Argo reference table 4     
_FillValue                  0  G�   DC_REFERENCE                  	long_name         (Station unique identifier in data centre   conventions       Data centre convention     
_FillValue                 �  G�   DATA_STATE_INDICATOR                  	long_name         1Degree of processing the data have passed through      conventions       Argo reference table 6     
_FillValue                  \  J�   	DATA_MODE                  	long_name         Delayed mode or real time data     conventions       >R : real time; D : delayed mode; A : real time with adjustment     
_FillValue                    K(   PLATFORM_TYPE                     	long_name         Type of float      conventions       Argo reference table 23    
_FillValue                 �  K@   FLOAT_SERIAL_NO                   	long_name         Serial number of the float     
_FillValue                 �  N    FIRMWARE_VERSION                  	long_name         Instrument firmware version    
_FillValue                 �  Q    WMO_INST_TYPE                     	long_name         Coded instrument type      conventions       Argo reference table 8     
_FillValue                  \  S�   JULD               	long_name         ?Julian day (UTC) of the station relative to REFERENCE_DATE_TIME    standard_name         time   units         "days since 1950-01-01 00:00:00 UTC     conventions       8Relative julian days with decimal part (as parts of day)   
resolution                   
_FillValue        A.�~       axis      T         �  T<   JULD_QC                	long_name         Quality on date and time   conventions       Argo reference table 2     
_FillValue                    T�   JULD_LOCATION                  	long_name         @Julian day (UTC) of the location relative to REFERENCE_DATE_TIME   units         "days since 1950-01-01 00:00:00 UTC     conventions       8Relative julian days with decimal part (as parts of day)   
resolution                   
_FillValue        A.�~          �  U   LATITUDE               	long_name         &Latitude of the station, best estimate     standard_name         latitude   units         degree_north   
_FillValue        @�i�       	valid_min         �V�        	valid_max         @V�        axis      Y         �  U�   	LONGITUDE                  	long_name         'Longitude of the station, best estimate    standard_name         	longitude      units         degree_east    
_FillValue        @�i�       	valid_min         �f�        	valid_max         @f�        axis      X         �  V|   POSITION_QC                	long_name         ,Quality on position (latitude and longitude)   conventions       Argo reference table 2     
_FillValue                    W4   POSITIONING_SYSTEM                    	long_name         Positioning system     
_FillValue                  �  WL   PROFILE_PRES_QC                	long_name         #Global quality flag of PRES profile    conventions       Argo reference table 2a    
_FillValue                    X   PROFILE_TEMP_QC                	long_name         #Global quality flag of TEMP profile    conventions       Argo reference table 2a    
_FillValue                    X   PROFILE_PSAL_QC                	long_name         #Global quality flag of PSAL profile    conventions       Argo reference table 2a    
_FillValue                    X4   VERTICAL_SAMPLING_SCHEME                  	long_name         Vertical sampling scheme   conventions       Argo reference table 16    
_FillValue                    XL   CONFIG_MISSION_NUMBER                  	long_name         :Unique number denoting the missions performed by the float     conventions       !1...N, 1 : first complete mission      
_FillValue         ��      \  oL   PRES         
      
   	long_name         )Sea water pressure, equals 0 at sea-level      standard_name         sea_water_pressure     
_FillValue        G�O�   units         decibar    	valid_min                	valid_max         F;�    C_format      %7.1f      FORTRAN_format        F7.1   
resolution        ?�     axis      Z          o�   PRES_QC          
         	long_name         quality flag   conventions       Argo reference table 2     
_FillValue                   ��   PRES_ADJUSTED            
      
   	long_name         )Sea water pressure, equals 0 at sea-level      standard_name         sea_water_pressure     
_FillValue        G�O�   units         decibar    	valid_min                	valid_max         F;�    C_format      %7.1f      FORTRAN_format        F7.1   
resolution        ?�     axis      Z          ��   PRES_ADJUSTED_QC         
         	long_name         quality flag   conventions       Argo reference table 2     
_FillValue                   ��   PRES_ADJUSTED_ERROR          
         	long_name         VContains the error on the adjusted values as determined by the delayed mode QC process     
_FillValue        G�O�   units         decibar    C_format      %7.1f      FORTRAN_format        F7.1   
resolution        ?�         ��   TEMP         
      	   	long_name         $Sea temperature in-situ ITS-90 scale   standard_name         sea_water_temperature      
_FillValue        G�O�   units         degree_Celsius     	valid_min         �      	valid_max         B      C_format      %9.3f      FORTRAN_format        F9.3   
resolution        :�o       ��   TEMP_QC          
         	long_name         quality flag   conventions       Argo reference table 2     
_FillValue                   �   TEMP_ADJUSTED            
      	   	long_name         $Sea temperature in-situ ITS-90 scale   standard_name         sea_water_temperature      
_FillValue        G�O�   units         degree_Celsius     	valid_min         �      	valid_max         B      C_format      %9.3f      FORTRAN_format        F9.3   
resolution        :�o       �   TEMP_ADJUSTED_QC         
         	long_name         quality flag   conventions       Argo reference table 2     
_FillValue                   �$   TEMP_ADJUSTED_ERROR          
         	long_name         VContains the error on the adjusted values as determined by the delayed mode QC process     
_FillValue        G�O�   units         degree_Celsius     C_format      %9.3f      FORTRAN_format        F9.3   
resolution        :�o       ,   PSAL         
      	   	long_name         Practical salinity     standard_name         sea_water_salinity     
_FillValue        G�O�   units         psu    	valid_min         @      	valid_max         B$     C_format      %9.3f      FORTRAN_format        F9.3   
resolution        :�o      @   PSAL_QC          
         	long_name         quality flag   conventions       Argo reference table 2     
_FillValue                  0T   PSAL_ADJUSTED            
      	   	long_name         Practical salinity     standard_name         sea_water_salinity     
_FillValue        G�O�   units         psu    	valid_min         @      	valid_max         B$     C_format      %9.3f      FORTRAN_format        F9.3   
resolution        :�o      6\   PSAL_ADJUSTED_QC         
         	long_name         quality flag   conventions       Argo reference table 2     
_FillValue                  Np   PSAL_ADJUSTED_ERROR          
         	long_name         VContains the error on the adjusted values as determined by the delayed mode QC process     
_FillValue        G�O�   units         psu    C_format      %9.3f      FORTRAN_format        F9.3   
resolution        :�o      Tx   	PARAMETER               	            	long_name         /List of parameters with calibration information    conventions       Argo reference table 3     
_FillValue                 P l�   SCIENTIFIC_CALIB_EQUATION               	            	long_name         'Calibration equation for this parameter    
_FillValue                 E  p�   SCIENTIFIC_CALIB_COEFFICIENT            	            	long_name         *Calibration coefficients for this equation     
_FillValue                 E  ��   SCIENTIFIC_CALIB_COMMENT            	            	long_name         .Comment applying to this parameter calibration     
_FillValue                 E  ��   SCIENTIFIC_CALIB_DATE               	             	long_name         Date of calibration    conventions       YYYYMMDDHHMISS     
_FillValue                 � ?�   HISTORY_INSTITUTION                      	long_name         "Institution which performed action     conventions       Argo reference table 4     
_FillValue                  \ C�   HISTORY_STEP                     	long_name         Step in data processing    conventions       Argo reference table 12    
_FillValue                  \ D    HISTORY_SOFTWARE                     	long_name         'Name of software which performed action    conventions       Institution dependent      
_FillValue                  \ D\   HISTORY_SOFTWARE_RELEASE                     	long_name         2Version/release of software which performed action     conventions       Institution dependent      
_FillValue                  \ D�   HISTORY_REFERENCE                        	long_name         Reference of database      conventions       Institution dependent      
_FillValue                 � E   HISTORY_DATE                      	long_name         #Date the history record was created    conventions       YYYYMMDDHHMISS     
_FillValue                 D J�   HISTORY_ACTION                       	long_name         Action performed on data   conventions       Argo reference table 7     
_FillValue                  \ L   HISTORY_PARAMETER                        	long_name         (Station parameter action is performed on   conventions       Argo reference table 3     
_FillValue                 p Lt   HISTORY_START_PRES                    	long_name          Start pressure action applied on   
_FillValue        G�O�   units         decibar       \ M�   HISTORY_STOP_PRES                     	long_name         Stop pressure action applied on    
_FillValue        G�O�   units         decibar       \ N@   HISTORY_PREVIOUS_VALUE                    	long_name         +Parameter/Flag previous value before action    
_FillValue        G�O�      \ N�   HISTORY_QCTEST                       	long_name         <Documentation of tests performed, tests failed (in hex form)   conventions       EWrite tests performed when ACTION=QCP$; tests failed when ACTION=QCF$      
_FillValue                 p N�Argo profile    3.1 1.2 19500101000000  20120309194738  20220601234707  1901589 1901589 1901589 1901589 1901589 1901589 1901589 1901589 1901589 1901589 1901589 1901589 1901589 1901589 1901589 1901589 1901589 1901589 1901589 1901589 1901589 1901589 1901589 US ARGO PROJECT                                                 US ARGO PROJECT                                                 US ARGO PROJECT                                                 US ARGO PROJECT                                                 US ARGO PROJECT                                                 US ARGO PROJECT                                                 US ARGO PROJECT                                                 US ARGO PROJECT                                                 US ARGO PROJECT                                                 US ARGO PROJECT                                                 US ARGO PROJECT                                                 US ARGO PROJECT                                                 US ARGO PROJECT                                                 US ARGO PROJECT                                                 US ARGO PROJECT                                                 US ARGO PROJECT                                                 US ARGO PROJECT                                                 US ARGO PROJECT                                                 US ARGO PROJECT                                                 US ARGO PROJECT                                                 US ARGO PROJECT                                                 US ARGO PROJECT                                                 US ARGO PROJECT                                                 BRECK OWENS                                                     BRECK OWENS                                                     BRECK OWENS                                                     BRECK OWENS                                                     BRECK OWENS                                                     BRECK OWENS                                                     BRECK OWENS                                                     BRECK OWENS                                                     BRECK OWENS                                                     BRECK OWENS                                                     BRECK OWENS                                                     BRECK OWENS                                                     BRECK OWENS                                                     BRECK OWENS                                                     BRECK OWENS                                                     BRECK OWENS                                                     BRECK OWENS                                                     BRECK OWENS                                                     BRECK OWENS                                                     BRECK OWENS                                                     BRECK OWENS                                                     BRECK OWENS                                                     BRECK OWENS                                                     PRES            TEMP            PSAL            PRES            TEMP            PSAL            PRES            TEMP            PSAL            PRES            TEMP            PSAL            PRES            TEMP            PSAL            PRES            TEMP            PSAL            PRES            TEMP            PSAL            PRES            TEMP            PSAL            PRES            TEMP            PSAL            PRES            TEMP            PSAL            PRES            TEMP            PSAL            PRES            TEMP            PSAL            PRES            TEMP            PSAL            PRES            TEMP            PSAL            PRES            TEMP            PSAL            PRES            TEMP            PSAL            PRES            TEMP            PSAL            PRES            TEMP            PSAL            PRES            TEMP            PSAL            PRES            TEMP            PSAL            PRES            TEMP            PSAL            PRES            TEMP            PSAL            PRES            TEMP            PSAL                                           	   
                                    AAAAAAAAAAAAAAAAAAAAAAA AOAOAOAOAOAOAOAOAOAOAOAOAOAOAOAOAOAOAOAOAOAOAO  4444_104057_000                 4444_104057_001                 4444_104057_002                 4444_104057_003                 4444_104057_004                 4444_104057_005                 4444_104057_006                 4444_104057_007                 4444_104057_008                 4444_104057_009                 4444_104057_010                 4444_104057_011                 4444_104057_012                 4444_104057_013                 4444_104057_014                 4444_104057_015                 4444_104057_016                 4444_104057_017                 4444_104057_018                 4444_104057_019                 4444_104057_020                 4444_104057_021                 4444_104057_022                 2C  2C  2C  2C  2C  2C  2C  2C  2C  2C  2C  2C  2C  2C  2C  2C  2C  2C  2C  2C  2C  2C  2C  DDDDDDDDDDDDDDDDDDDDDDD SOLO_W                          SOLO_W                          SOLO_W                          SOLO_W                          SOLO_W                          SOLO_W                          SOLO_W                          SOLO_W                          SOLO_W                          SOLO_W                          SOLO_W                          SOLO_W                          SOLO_W                          SOLO_W                          SOLO_W                          SOLO_W                          SOLO_W                          SOLO_W                          SOLO_W                          SOLO_W                          SOLO_W                          SOLO_W                          SOLO_W                          1127                            1127                            1127                            1127                            1127                            1127                            1127                            1127                            1127                            1127                            1127                            1127                            1127                            1127                            1127                            1127                            1127                            1127                            1127                            1127                            1127                            1127                            1127                            2.06                            2.06                            2.06                            2.06                            2.06                            2.06                            2.06                            2.06                            2.06                            2.06                            2.06                            2.06                            2.06                            2.06                            2.06                            2.06                            2.06                            2.06                            2.06                            2.06                            2.06                            2.06                            2.06                            851 851 851 851 851 851 851 851 851 851 851 851 851 851 851 851 851 851 851 851 851 851 851 @�-$��ax@�/d�ò�@�1��K�@�4d�ޠ@�6�}� �@�9hE6�@�;�+��@�>g�o��@�@�\(�@�Cva�P�@�E�d�
@�Hvq�r@�J��m�@�Mt5��%@�O�4�@�Ruw�	@�T�JV@�Wu�c��@�Y����@�\s���P@�^���/@�asȠ�Q@�c����11111111111111111111111 @�-$�www@�/eS?V@�1���Y@�4iPg(�@�6� ��@�9h�-��@�;膣�@�>hA��u@�@��C �@�Cvy��@�E��i�@�Hv��0*@�J�ߒ��@�Mt_#E@�O���@�Ru��%@�T���@�Wu��5�@�Y�*I��@�\w����@�^���ܻ@�at���@�c��|���I�^5?}��O�;d����O�;d���E������O�;dZ��Q����7KƧ⟾vȴ9��`A�7Kǿ�E�������+I�?�z�G�{��p��
=q����"��`��z�G�����vȴ������+��1&�xտܼj~��#�۶E�������$�/�߮z�G���      �3�|�hs�3�� ě��3�n��O��3���l�D�3����l��3&�x���3S�E����3�\(��3�`A�7L�3�M����3Η�O�;�3d���S��2��1'�3 ě��T�2���v��2��+�2���`A��2�t�j~��2qhr� ��2DZ�1�2���"���2�ȴ9X�2["��`B11111111111111111111111 ARGOS   ARGOS   ARGOS   ARGOS   ARGOS   ARGOS   ARGOS   ARGOS   ARGOS   ARGOS   ARGOS   ARGOS   ARGOS   ARGOS   ARGOS   ARGOS   ARGOS   ARGOS   ARGOS   ARGOS   ARGOS   ARGOS   ARGOS   AAAAAAAAAAAAAAAAAAAAAAA AAAAAAAAAAAAAAAAAAAAAAA AAAAAAAAAAAAAFFAAAAAAAA Primary sampling: averaged [data averaged with equal weights into irregular pressure bins                                                                                                                                                                       Primary sampling: averaged [data averaged with equal weights into irregular pressure bins                                                                                                                                                                       Primary sampling: averaged [data averaged with equal weights into irregular pressure bins                                                                                                                                                                       Primary sampling: averaged [data averaged with equal weights into irregular pressure bins                                                                                                                                                                       Primary sampling: averaged [data averaged with equal weights into irregular pressure bins                                                                                                                                                                       Primary sampling: averaged [data averaged with equal weights into irregular pressure bins                                                                                                                                                                       Primary sampling: averaged [data averaged with equal weights into irregular pressure bins                                                                                                                                                                       Primary sampling: averaged [data averaged with equal weights into irregular pressure bins                                                                                                                                                                       Primary sampling: averaged [data averaged with equal weights into irregular pressure bins                                                                                                                                                                       Primary sampling: averaged [data averaged with equal weights into irregular pressure bins                                                                                                                                                                       Primary sampling: averaged [data averaged with equal weights into irregular pressure bins                                                                                                                                                                       Primary sampling: averaged [data averaged with equal weights into irregular pressure bins                                                                                                                                                                       Primary sampling: averaged [data averaged with equal weights into irregular pressure bins                                                                                                                                                                       Primary sampling: averaged [data averaged with equal weights into irregular pressure bins                                                                                                                                                                       Primary sampling: averaged [data averaged with equal weights into irregular pressure bins                                                                                                                                                                       Primary sampling: averaged [data averaged with equal weights into irregular pressure bins                                                                                                                                                                       Primary sampling: averaged [data averaged with equal weights into irregular pressure bins                                                                                                                                                                       Primary sampling: averaged [data averaged with equal weights into irregular pressure bins                                                                                                                                                                       Primary sampling: averaged [data averaged with equal weights into irregular pressure bins                                                                                                                                                                       Primary sampling: averaged [data averaged with equal weights into irregular pressure bins                                                                                                                                                                       Primary sampling: averaged [data averaged with equal weights into irregular pressure bins                                                                                                                                                                       Primary sampling: averaged [data averaged with equal weights into irregular pressure bins                                                                                                                                                                       Primary sampling: averaged [data averaged with equal weights into irregular pressure bins                                                                                                                                                                                                                                            @�  A   Ap  A�  A�  A�  B  B   B4  BH  B\  Bp  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  C  C  C  C  C  C  C   C%  C*  C/  C4  C9  C>  CC  CH  C\  Cp  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  D	� D  D"� D/  D;� DH  DT� Da  Dm� Dz  D�@ D�� D�� D�  D�@ @�  A   Ap  A�  A�  A�  B  B   B4  BH  B\  Bp  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  C  C  C  C  C  C  C   C%  C*  C/  C4  C9  C>  CC  CH  C\  Cp  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  D	� D  D"� D/  D;� DH  DT� Da  Dm� Dz  D�@ D�� D�� D�  G�O�@�  A   Ap  A�  A�  A�  B  B   B4  BH  B\  Bp  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  C  C  C  C  C  C  C   C%  C*  C/  C4  C9  C>  CC  CH  C\  Cp  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  D	� D  D"� D/  D;� DH  DT� Da  Dm� Dz  D�@ D�� D�� D�  D�@ @�  A   Ap  A�  A�  A�  B  B   B4  BH  B\  Bp  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  C  C  C  C  C  C  C   C%  C*  C/  C4  C9  C>  CC  CH  C\  Cp  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  D	� D  D"� D/  D;� DH  DT� Da  Dm� Dz  D�@ D�� D�� D�  G�O�@�  A   Ap  A�  A�  A�  B  B   B4  BH  B\  Bp  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  C  C  C  C  C  C  C   C%  C*  C/  C4  C9  C>  CC  CH  C\  Cp  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  D	� D  D"� D/  D;� DH  DT� Da  Dm� Dz  D�@ D�� D�� D�  D�@ @�  A   Ap  A�  A�  A�  B  B   B4  BH  B\  Bp  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  C  C  C  C  C  C  C   C%  C*  C/  C4  C9  C>  CC  CH  C\  Cp  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  D	� D  D"� D/  D;� DH  DT� Da  Dm� Dz  D�@ D�� D�� D�  D�@ @�  A   Ap  A�  A�  A�  B  B   B4  BH  B\  Bp  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  C  C  C  C  C  C  C   C%  C*  C/  C4  C9  C>  CC  CH  C\  Cp  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  D	� D  D"� D/  D;� DH  DT� Da  Dm� Dz  D�@ D�� D�� D�  D�@ @�  A   Ap  A�  A�  A�  B  B   B4  BH  B\  Bp  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  C  C  C  C  C  C  C   C%  C*  C/  C4  C9  C>  CC  CH  C\  Cp  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  D	� D  D"� D/  D;� DH  DT� Da  Dm� Dz  D�@ D�� D�� D�  D�@ @�  A   Ap  A�  A�  A�  B  B   B4  BH  B\  Bp  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  C  C  C  C  C  C  C   C%  C*  C/  C4  C9  C>  CC  CH  C\  Cp  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  D	� D  D"� D/  D;� DH  DT� Da  Dm� Dz  D�@ D�� D�� D�  D�@ @�  A   Ap  A�  A�  A�  B  B   B4  BH  B\  Bp  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  C  C  C  C  C  C  C   C%  C*  C/  C4  C9  C>  CC  CH  C\  Cp  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  D	� D  D"� D/  D;� DH  DT� Da  Dm� Dz  D�@ D�S3G�O�G�O�G�O�@�  A   Ap  A�  A�  A�  B  B   B4  BH  B\  Bp  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  C  C  C  C  C  C  C   C%  C*  C/  C4  C9  C>  CC  CH  C\  Cp  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  D	� D  D"� D/  D;� DH  DT� Da  Dm� Dz  D�@ D�vfG�O�G�O�G�O�@�  A   Ap  A�  A�  A�  B  B   B4  BH  B\  Bp  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  C  C  C  C  C  C  C   C%  C*  C/  C4  C9  C>  CC  CH  C\  Cp  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  D	� D  D"� D/  D;� DH  DT� Da  Dm� Dz  D�@ G�O�G�O�G�O�G�O�@�  A   Ap  A�  A�  A�  B  B   B4  BH  B\  Bp  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  C  C  C  C  C  C  C   C%  C*  C/  C4  C9  C>  CC  CH  C\  Cp  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  D	� D  D"� D/  D;� DH  DT� Da  Dm� Dz  D�@ G�O�G�O�G�O�G�O�@�  A   Ap  A�  A�  A�  B  B   B4  BH  B\  Bp  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  C  C  C  C  C  C  C   C%  C*  C/  C4  C9  C>  CC  CH  C\  Cp  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  D	� D  D"� D/  D;� DH  DT� Da  Dm� Dz  G�O�G�O�G�O�G�O�G�O�@�  A   Ap  A�  A�  A�  B  B   B4  BH  B\  Bp  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  C  C  C  C  C  C  C   C%  C*  C/  C4  C9  C>  CC  CH  C\  Cp  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  D	� D  D"� D/  D;� DH  DT� Da  Dm� Dz  G�O�G�O�G�O�G�O�G�O�@�  A   Ap  A�  A�  A�  B  B   B4  BH  B\  Bp  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  C  C  C  C  C  C  C   C%  C*  C/  C4  C9  C>  CC  CH  C\  Cp  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  D	� D  D"� D/  D;� DH  DT� Da  Dm� Dz  D�@ G�O�G�O�G�O�G�O�@�  A   Ap  A�  A�  A�  B  B   B4  BH  B\  Bp  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  C  C  C  C  C  C  C   C%  C*  C/  C4  C9  C>  CC  CH  C\  Cp  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  D	� D  D"� D/  D;� DH  DT� Da  Dm� Dz  G�O�G�O�G�O�G�O�G�O�@�  A   Ap  A�  A�  A�  B  B   B4  BH  B\  Bp  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  C  C  C  C  C  C  C   C%  C*  C/  C4  C9  C>  CC  CH  C\  Cp  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  D	� D  D"� D/  D;� DH  DT� Da  Dm� Dz  G�O�G�O�G�O�G�O�G�O�@�  A   Ap  A�  A�  A�  B  B   B4  BH  B\  Bp  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  C  C  C  C  C  C  C   C%  C*  C/  C4  C9  C>  CC  CH  C\  Cp  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  D	� D  D"� D/  D;� DH  DT� Da  Dm� Dz  G�O�G�O�G�O�G�O�G�O�@�  A   Ap  A�  A�  A�  B  B   B4  BH  B\  Bp  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  C  C  C  C  C  C  C   C%  C*  C/  C4  C9  C>  CC  CH  C\  Cp  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  D	� D  D"� D/  D;� DH  DT� Da  Dm� Dz  G�O�G�O�G�O�G�O�G�O�@�  A   Ap  A�  A�  A�  B  B   B4  BH  B\  Bp  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  C  C  C  C  C  C  C   C%  C*  C/  C4  C9  C>  CC  CH  C\  Cp  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  D	� D  D"� D/  D;� DH  DT� Da  Dm� Dz  G�O�G�O�G�O�G�O�G�O�@�  A   Ap  A�  A�  A�  B  B   B4  BH  B\  Bp  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  C  C  C  C  C  C  C   C%  C*  C/  C4  C9  C>  CC  CH  C\  Cp  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  D	� D  D"� D/  D;� DH  DT� Da  Dm� Dz  G�O�G�O�G�O�G�O�G�O�@�  A   Ap  A�  A�  A�  B  B   B4  BH  B\  Bp  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  C  C  C  C  C  C  C   C%  C*  C/  C4  C9  C>  CC  CH  C\  Cp  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  D	� D  D"� D/  D;� DH  DT� Da  Dm� Dz  G�O�G�O�G�O�G�O�G�O�1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111 1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111 111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111   1111111111111111111111111111111111111111111111111111111111111111   111111111111111111111111111111111111111111111111111111111111111    111111111111111111111111111111111111111111111111111111111111111    11111111111111111111111111111111111111111111111111111111111111     11111111111111111111111111111111111111111111111111111111111111     111111111111111111111111111111111111111111111111111111111111111    11111111111111111111111111111111111111111111111111111111111111     11111111111111111111111111111111111111111111111111111111111111     11111111111111111111111111111111111111111111111111111111111111     11111111111111111111111111111111111111111111111111111111111111     11111111111111111111111111111111111111111111111111111111111111     11111111111111111111111111111111111111111111111111111111111111     11111111111111111111111111111111111111111111111111111111111111        @�  A   Ap  A�  A�  A�  B  B   B4  BH  B\  Bp  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  C  C  C  C  C  C  C   C%  C*  C/  C4  C9  C>  CC  CH  C\  Cp  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  D	� D  D"� D/  D;� DH  DT� Da  Dm� Dz  D�@ D�� D�� D�  D�@ @�  A   Ap  A�  A�  A�  B  B   B4  BH  B\  Bp  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  C  C  C  C  C  C  C   C%  C*  C/  C4  C9  C>  CC  CH  C\  Cp  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  D	� D  D"� D/  D;� DH  DT� Da  Dm� Dz  D�@ D�� D�� D�  G�O�@�  A   Ap  A�  A�  A�  B  B   B4  BH  B\  Bp  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  C  C  C  C  C  C  C   C%  C*  C/  C4  C9  C>  CC  CH  C\  Cp  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  D	� D  D"� D/  D;� DH  DT� Da  Dm� Dz  D�@ D�� D�� D�  D�@ @�  A   Ap  A�  A�  A�  B  B   B4  BH  B\  Bp  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  C  C  C  C  C  C  C   C%  C*  C/  C4  C9  C>  CC  CH  C\  Cp  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  D	� D  D"� D/  D;� DH  DT� Da  Dm� Dz  D�@ D�� D�� D�  G�O�@�  A   Ap  A�  A�  A�  B  B   B4  BH  B\  Bp  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  C  C  C  C  C  C  C   C%  C*  C/  C4  C9  C>  CC  CH  C\  Cp  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  D	� D  D"� D/  D;� DH  DT� Da  Dm� Dz  D�@ D�� D�� D�  D�@ @�  A   Ap  A�  A�  A�  B  B   B4  BH  B\  Bp  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  C  C  C  C  C  C  C   C%  C*  C/  C4  C9  C>  CC  CH  C\  Cp  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  D	� D  D"� D/  D;� DH  DT� Da  Dm� Dz  D�@ D�� D�� D�  D�@ @�  A   Ap  A�  A�  A�  B  B   B4  BH  B\  Bp  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  C  C  C  C  C  C  C   C%  C*  C/  C4  C9  C>  CC  CH  C\  Cp  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  D	� D  D"� D/  D;� DH  DT� Da  Dm� Dz  D�@ D�� D�� D�  D�@ @�  A   Ap  A�  A�  A�  B  B   B4  BH  B\  Bp  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  C  C  C  C  C  C  C   C%  C*  C/  C4  C9  C>  CC  CH  C\  Cp  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  D	� D  D"� D/  D;� DH  DT� Da  Dm� Dz  D�@ D�� D�� D�  D�@ @�  A   Ap  A�  A�  A�  B  B   B4  BH  B\  Bp  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  C  C  C  C  C  C  C   C%  C*  C/  C4  C9  C>  CC  CH  C\  Cp  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  D	� D  D"� D/  D;� DH  DT� Da  Dm� Dz  D�@ D�� D�� D�  D�@ @�  A   Ap  A�  A�  A�  B  B   B4  BH  B\  Bp  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  C  C  C  C  C  C  C   C%  C*  C/  C4  C9  C>  CC  CH  C\  Cp  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  D	� D  D"� D/  D;� DH  DT� Da  Dm� Dz  D�@ D�S3G�O�G�O�G�O�@�  A   Ap  A�  A�  A�  B  B   B4  BH  B\  Bp  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  C  C  C  C  C  C  C   C%  C*  C/  C4  C9  C>  CC  CH  C\  Cp  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  D	� D  D"� D/  D;� DH  DT� Da  Dm� Dz  D�@ D�vfG�O�G�O�G�O�@�  A   Ap  A�  A�  A�  B  B   B4  BH  B\  Bp  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  C  C  C  C  C  C  C   C%  C*  C/  C4  C9  C>  CC  CH  C\  Cp  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  D	� D  D"� D/  D;� DH  DT� Da  Dm� Dz  D�@ G�O�G�O�G�O�G�O�@�  A   Ap  A�  A�  A�  B  B   B4  BH  B\  Bp  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  C  C  C  C  C  C  C   C%  C*  C/  C4  C9  C>  CC  CH  C\  Cp  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  D	� D  D"� D/  D;� DH  DT� Da  Dm� Dz  D�@ G�O�G�O�G�O�G�O�@�  A   Ap  A�  A�  A�  B  B   B4  BH  B\  Bp  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  C  C  C  C  C  C  C   C%  C*  C/  C4  C9  C>  CC  CH  C\  Cp  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  D	� D  D"� D/  D;� DH  DT� Da  Dm� Dz  G�O�G�O�G�O�G�O�G�O�@�  A   Ap  A�  A�  A�  B  B   B4  BH  B\  Bp  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  C  C  C  C  C  C  C   C%  C*  C/  C4  C9  C>  CC  CH  C\  Cp  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  D	� D  D"� D/  D;� DH  DT� Da  Dm� Dz  G�O�G�O�G�O�G�O�G�O�@�  A   Ap  A�  A�  A�  B  B   B4  BH  B\  Bp  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  C  C  C  C  C  C  C   C%  C*  C/  C4  C9  C>  CC  CH  C\  Cp  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  D	� D  D"� D/  D;� DH  DT� Da  Dm� Dz  D�@ G�O�G�O�G�O�G�O�@�  A   Ap  A�  A�  A�  B  B   B4  BH  B\  Bp  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  C  C  C  C  C  C  C   C%  C*  C/  C4  C9  C>  CC  CH  C\  Cp  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  D	� D  D"� D/  D;� DH  DT� Da  Dm� Dz  G�O�G�O�G�O�G�O�G�O�@�  A   Ap  A�  A�  A�  B  B   B4  BH  B\  Bp  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  C  C  C  C  C  C  C   C%  C*  C/  C4  C9  C>  CC  CH  C\  Cp  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  D	� D  D"� D/  D;� DH  DT� Da  Dm� Dz  G�O�G�O�G�O�G�O�G�O�@�  A   Ap  A�  A�  A�  B  B   B4  BH  B\  Bp  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  C  C  C  C  C  C  C   C%  C*  C/  C4  C9  C>  CC  CH  C\  Cp  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  D	� D  D"� D/  D;� DH  DT� Da  Dm� Dz  G�O�G�O�G�O�G�O�G�O�@�  A   Ap  A�  A�  A�  B  B   B4  BH  B\  Bp  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  C  C  C  C  C  C  C   C%  C*  C/  C4  C9  C>  CC  CH  C\  Cp  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  D	� D  D"� D/  D;� DH  DT� Da  Dm� Dz  G�O�G�O�G�O�G�O�G�O�@�  A   Ap  A�  A�  A�  B  B   B4  BH  B\  Bp  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  C  C  C  C  C  C  C   C%  C*  C/  C4  C9  C>  CC  CH  C\  Cp  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  D	� D  D"� D/  D;� DH  DT� Da  Dm� Dz  G�O�G�O�G�O�G�O�G�O�@�  A   Ap  A�  A�  A�  B  B   B4  BH  B\  Bp  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  C  C  C  C  C  C  C   C%  C*  C/  C4  C9  C>  CC  CH  C\  Cp  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  D	� D  D"� D/  D;� DH  DT� Da  Dm� Dz  G�O�G�O�G�O�G�O�G�O�@�  A   Ap  A�  A�  A�  B  B   B4  BH  B\  Bp  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  C  C  C  C  C  C  C   C%  C*  C/  C4  C9  C>  CC  CH  C\  Cp  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  D	� D  D"� D/  D;� DH  DT� Da  Dm� Dz  G�O�G�O�G�O�G�O�G�O�1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111 1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111 111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111   1111111111111111111111111111111111111111111111111111111111111111   111111111111111111111111111111111111111111111111111111111111111    111111111111111111111111111111111111111111111111111111111111111    11111111111111111111111111111111111111111111111111111111111111     11111111111111111111111111111111111111111111111111111111111111     111111111111111111111111111111111111111111111111111111111111111    11111111111111111111111111111111111111111111111111111111111111     11111111111111111111111111111111111111111111111111111111111111     11111111111111111111111111111111111111111111111111111111111111     11111111111111111111111111111111111111111111111111111111111111     11111111111111111111111111111111111111111111111111111111111111     11111111111111111111111111111111111111111111111111111111111111     11111111111111111111111111111111111111111111111111111111111111        @��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��G�O�@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��G�O�@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��G�O�G�O�G�O�@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��G�O�G�O�G�O�@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��G�O�G�O�G�O�G�O�@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��G�O�G�O�G�O�G�O�@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��G�O�G�O�G�O�G�O�G�O�@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��G�O�G�O�G�O�G�O�G�O�@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��G�O�G�O�G�O�G�O�@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��G�O�G�O�G�O�G�O�G�O�@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��G�O�G�O�G�O�G�O�G�O�@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��G�O�G�O�G�O�G�O�G�O�@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��G�O�G�O�G�O�G�O�G�O�@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��G�O�G�O�G�O�G�O�G�O�@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��G�O�G�O�G�O�G�O�G�O�@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��@��G�O�G�O�G�O�G�O�G�O�A���AھwAڇ+A�VA��A�p�A�z�A�O�A�VA���AσA�G�A�$�AŮA�ffA���A���A�9XA���A�C�A�A|ȴAy��AtĜAnjAj�Ah��Af�AdE�Aa&�A_�A^ĜA]��A];dA\��A[VAX�+AV^5AV9XAU��AR �AKhsAA��A=�;A6�\A-�A$bNA�PAoA  A�j@�v�@�bN@�5?@�ƨ@��D@�X@���@�?}@�7L@�ȴ@�b@��D@��`@���@���@�JA�E�A���A�A݃A�(�A���A�
=A٬A�S�A���A�z�A�9XA�XA�z�A��A���A�jA�l�A�=qAK�Av�RAq�;Ag�
A`ȴA_G�A^A�A[
=AYO�AW&�ATĜATAS�7AS?}AQ�AP�\ANbNAL�AKO�AJ1'AI�wAEK�AA�A933A1&�A,�jA+33A#�A"�AI�AI�@��@�ȴ@��@���@���@��@��!@��9@�l�@���@�=q@���@�(�@���@��@���G�O�A�\)A���A۲-AہA�5?Aں^A��A�l�A�r�A��A���A�A�33A��mA�oA���A�r�A���As�Ai�AdM�A_�PA^ffA\�A[S�AZ  AY`BAW�AV-AU�
AU%AT �ASXAS"�AR�AR��AR�+AR�AQ�TAQ��AO�AH$�A9+A(Q�A#t�A�A�FAdZA-A	|�@��@ו�@�p�@���@��@��j@���@��@���@��@�r�@�;d@��@�%@��h@��@�JA���Aؙ�A�~�A�bNA�;dA��A�p�AָRA�dZA�"�A�hsA��A�S�A~��Ay�#Au�TAn�Al$�Aj^5Ah�AgVAfZAe�Aa�A^��A]��A\�/A\^5A\�A[��A[�AZ��AY�TAX(�AW\)AU�AU�hAU%ATZAT  AQ&�AN�uAKXAFjA;K�A5VA(�HA�A�A	��@�O�@�;d@�@�@��@��h@��T@�=q@��@��@�~�@�ff@�A�@���@�?}@���G�O�AڶFA�jA�33A��TAٍPA��yA�
=A֧�A�S�A� �A�v�A�|�A��;A��A�C�A��A|ȴAw`BAsdZAo�TAl�/Ak�Ai��Ag��Ad�!A`(�A^��A^�A]x�A\�\A[��A[7LAZ�HAZ9XAY�PAYXAYAXVAW�hAV��APAE7LA9�^A2$�A/�mA+�-A%G�A"Q�A��Ahs@��P@�"�@�E�@�@�Ĝ@���@�Z@�b@�n�@��9@�n�@�dZ@�A�@�Ĝ@�/@��7@���Aܥ�A܇+A�ffA�A�A���A�=qAڕ�A�l�A�&�A���A��wA���A�A��TA�ȴA�l�A�\)A��7A�&�A� �A�;dAy%AtE�ArffAn�DAi�Af(�Ad�/Ad^5Ab�A_"�A^5?A^�A]A]\)A]/A\�A[�mAZQ�AY%AL��AFA�A;t�A1��A*�+A"�\A�-A�;AȴA�A�9@߾w@��@�{@��@� �@���@��9@��@��
@��y@�o@��@��@�&�@���@��7A��A��yA�jA�5?Aء�A���A���AԁA�jAĴ9A�
=A��A�A�A���A�%A�7LA�z�A|jAw��As��Ao�;Am+Ai�^AfbAd��AdJAc/A`��A_`BA]�;A]O�A]"�A]�A]%A\�A\��A\�+A\1AZ�yAZ1'AVz�AM;dAAdZA,�\A&��A!"�A�wAȴA1A��@���@�x�@�Q�@�x�@��@���@�X@�@��7@�
=@�M�@�|�@��;@���@�G�@��^@��#A�S�A�p�AГuA�AΡ�A��A���A�-A�A�5?A��A�A�9XA�jA�^5A�$�A��\A�AydZAx�uAx(�Aw�AtE�Ar{Ap��AoC�AnffAk�;AjE�Ai��Ae��A_��A^ �A\r�A[�^A[x�A[?}A[VAZ�/AZ~�AP��AG��A@�yA3�TA,��A& �A�A�AA�A��A�w@�C�@�G�@��@�  @�-@�l�@�G�@� �@�ff@�;d@�33@��@���@�7L@�x�@��7A��/A��AҸRA��A��A�ȴA��
A�9XA��#A��`A�A��;A���A��;A�"�A�VA���A|9XA{�wAz�Ay�
AxffAwAvI�At��As;dArApv�Am�
Aj�DAf1'A_�^A]&�A\�DA\9XA\1A[��AZv�AY��AYXAQ�7AF��A:��A(�uA�A��A�DA+A+A	/A�@��y@�A�@���@���@�-@�+@�~�@�K�@���@�33@�  @��9@�V@�hs@���@���A��HA΋DA��A��TA�bNA��/A�A�A�r�A�VA� �A�;dA�%A�-A��A�7LA��PA��`A�ZA��A��A}��AxQ�As��Ar�!Ap-Anr�Amx�Al�Aj�Ait�Af��AedZAd�Ab�A`�A_S�A]S�A\��A\�\A\ffAT�9AIXA=x�A3A)��A"{A�hA��A"�A
�A bN@���@�@��@���@��@�&�@�{@�E�@�Q�@���@�r�@���@��G�O�G�O�G�O�A���A�1A��;A���A˼jA��A��`A�n�A��A��A��A��uA�ƨA�33A� �A�%A�S�A��DA�1A�S�A|v�Ay/Av�+AsS�An�uAk�TAj1Ai"�Ag�mAg33AfbNAe;dAd�Ab�AaXA^�DA\��A[�A[XA[;dAWC�AI�#A<ZA9��A.��A ��AffAC�A��A�@�5?@���@��@��@�I�@��
@�&�@�`B@�Z@�7L@��@�`B@��@��G�O�G�O�G�O�A�/A��A�=qA�7LA���A�O�A�I�A�  A��A���A�Q�A���A�&�A�G�A�
=A{&�Ay33AxE�AxbAu/Aqt�Ao��Al��Aj��Ai��Ah��Ah�Ag�mAf�Af�Ae�Ae�mAe�;Ae��Ae�Aap�A_��A^v�A]hsA[VAXȴAUƨAJ �A2�A'�A!�AA�A`BAJ@��w@�9@��@�bN@��@��R@��+@�-@�9X@�I�@�M�@���@�?}G�O�G�O�G�O�G�O�A؅A�A�n�AԲ-A�v�A�n�A���A���A�;dA� �A�/A�t�A�VA��-A���A�A�JA�+A��A���A}�^A{��A{t�A{AvA�Ap9XAkl�Ai%Ag�
AgdZAe�-Ad�Ad=qAb��AaO�A`n�A_��A^�\A]33A\VAZ��AR~�AA��A<�A6n�A*�A!S�A�A�TA
�`A%@�Z@���@�-@���@�%@��@�I�@�O�@��@��/@���@��G�O�G�O�G�O�G�O�A�7LAøRA�;dA�x�A��A���A��A��A�r�A��HA�oA���A�|�A�G�A}�A{"�Aw�;As�Am�-Al�jAk�Ai
=Af��Ad��AcAa��A`�RA^��A\�`A[dZAZĜAZ�DAZffAZQ�AZA�AZ5?AZ1AY�^AX�jAWAO�AI��AE|�A?%A1��A!K�A5?A33A
=AO�A�F@�9X@��m@���@�
=@�"�@��@�1@�b@�5?@� �@��`G�O�G�O�G�O�G�O�G�O�A�C�A�5?A��uA�E�A�bA��9A�G�A�A��-A��A�-A�G�A���A� �A�~�A� �A~��Ay��At��An��Aj��Ah�jAe�A^�DA["�AZ��AZbNAZM�AZI�AZI�AZA�AZ1'AZ�AY�AY�7AX��AWƨAU%AQp�AMp�AHĜAEƨA4JA ��A�HAZA�\A�7A\)A�!@�ff@�@� �@�dZ@�"�@�@���@��j@��R@�/@���@�dZG�O�G�O�G�O�G�O�G�O�A���A� �A��mA��A�n�A�&�A��`A�~�A��A��DA�A��A�O�A�S�A��^A�33A�ZA��A�dZA�FA}��A|bNAz~�Ax�`Aw�-At(�Aox�Aj��Ai�Ah��AhȴAh�RAh  Ae\)Abn�A`=qA^ZA\5?A[�PA[`BAS��AJZAH��ADI�A1�A#VA�TA(�A	;dAt�@���@�{@�A�@���@���@�p�@�9X@��@��h@�dZ@�?}@���@��G�O�G�O�G�O�G�O�APA�\)A��yA�bA���A���A�S�A�1'A�jA���A��wA�{A��#A�hsA��A��!A��7A�VA���A�%A��A��TA�ffAs�Al��Ad�`Ad �Ac��Acl�Ab{A_�A\��A\��A\E�A[��AZJAX5?AW��AWAU��AN�ALȴAEG�A21'A*��A?}AdZA�TA|�A~�@�bN@��@��y@�hs@��`@�x�@�E�@�K�@�{@�r�@��\@���G�O�G�O�G�O�G�O�G�O�A��A���A�M�A�M�A���A�VA��PA�bNA�ƨA�ĜA�-A���A�ZA�7LA��;A�bA���A���A���A��A�l�A�bA�^5A�jA���A���A}?}AyK�ArVAh�DAc�AbVAa��A`JA_7LA^ȴA]��A[�AYoAX1'AO�PAK�mA<r�A45?A-p�A��AhsA�\A�A1@��@�J@��@�ƨ@�Z@�@�O�@���@���@�  @�ff@�n�G�O�G�O�G�O�G�O�G�O�A��`A�A�A�~�A��A��A�|�A��A�$�A��A�A��A���A�ƨA��\A��RA���A�`BA���A���A� �A�A��A|��Azr�Ax-Av(�As�Am
=AiAf��AeVAb�A`jA\�jA[t�A[+AZ�AZz�AY��AV �AQp�AO�AM;dAAVA7hsA.��A#dZA5?A�A%A+@�`B@և+@̛�@� �@��y@���@�\)@��F@��@�b@���G�O�G�O�G�O�G�O�G�O�A�/A��AčPAÍPA��mA�JA�{A�VA�`BA��A�`BA�/A�  A�x�A� �A���A��7A��A���A�n�A��!A�Q�A���A�JA�t�Av�RAnbNAk�Ag?}Acl�A`-A^��A\��A[�^A[oAZ^5AZ-AY�AY�hAX�\ARz�AN�!AM��AC�A)p�AS�AoAG�A�;A��@���@�bN@�I�@��`@�\)@���@��h@�dZ@��@�Ĝ@��u@�{G�O�G�O�G�O�G�O�G�O�A��A���A�G�AƉ7A�t�A�hsA�^5A�K�A�7LA�{A���AļjA�`BA�Q�A��A���A���A���A���A�%A��mA��A�E�A{��AuhsAp  Ak��Ac�A`�+A`=qA_�A_�A_dZA_/A^��A]�;A]�hA]l�A]O�A\�`A[\)AT��AH$�A;+A3S�A+�A =qAȴA1A��@��-@�O�@ҸR@�?}@���@���@�V@���@��+@�r�@��7@��/G�O�G�O�G�O�G�O�G�O�A�S�A���A���A���Aʙ�A�jA�9XA���Aɗ�A�
=A�hsA���A�Q�A��A��A�+A���A�{A���A�x�A��A�VA���Az��Ay�PAwXAu�FAsl�Aq�;Ap=qAkAg��Af�9AeAe�Ab�yAap�A`~�A_��A^��AW|�ARJAK�7A?�FA4ffA+�mA��AoA�A��@�Z@ᙚ@��@��@���@��+@���@��@�O�@�j@�x�@���G�O�G�O�G�O�G�O�G�O�A�7LA�33A�&�A�A�oA���A��A��A���A͟�A�ZA�JAˁA�E�A��A�|�A���A���Aux�Aq��An��Al��Al�RAl �AjQ�Ah��Af�DAdAb��A`�uA^�jA];dA[�A["�AY�FAXȴAW��AW�AV�+AUG�AP��AM�PAG�A8ĜA5��A)G�A�9A�#A�AG�@��
@��T@��@��/@���@�"�@��@�Q�@�l�@��@�`B@�r�G�O�G�O�G�O�G�O�G�O�1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111 1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111 111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111   1111111111111111111111111111111111111111111111111111111111111111   111111111111111111111111111111111111111111111111111111111111111    111111111111111111111111111111111111111111111111111111111111111    11111111111111111111111111111111111111111111111111111111111111     11111111111111111111111111111111111111111111111111111111111111     111111111111111111111111111111111111111111111111111111111111111    11111111111111111111111111111111111111111111111111111111111111     11111111111111111111111111111111111111111111111111111111111111     11111111111111111111111111111111111111111111111111111111111111     11111111111111111111111111111111111111111111111111111111111111     11111111111111111111111111111111111111111111111111111111111111     11111111111111111111111111111111111111111111111111111111111111     11111111111111111111111111111111111111111111111111111111111111        A���AھwAڇ+A�VA��A�p�A�z�A�O�A�VA���AσA�G�A�$�AŮA�ffA���A���A�9XA���A�C�A�A|ȴAy��AtĜAnjAj�Ah��Af�AdE�Aa&�A_�A^ĜA]��A];dA\��A[VAX�+AV^5AV9XAU��AR �AKhsAA��A=�;A6�\A-�A$bNA�PAoA  A�j@�v�@�bN@�5?@�ƨ@��D@�X@���@�?}@�7L@�ȴ@�b@��D@��`@���@���@�JA�E�A���A�A݃A�(�A���A�
=A٬A�S�A���A�z�A�9XA�XA�z�A��A���A�jA�l�A�=qAK�Av�RAq�;Ag�
A`ȴA_G�A^A�A[
=AYO�AW&�ATĜATAS�7AS?}AQ�AP�\ANbNAL�AKO�AJ1'AI�wAEK�AA�A933A1&�A,�jA+33A#�A"�AI�AI�@��@�ȴ@��@���@���@��@��!@��9@�l�@���@�=q@���@�(�@���@��@���G�O�A�\)A���A۲-AہA�5?Aں^A��A�l�A�r�A��A���A�A�33A��mA�oA���A�r�A���As�Ai�AdM�A_�PA^ffA\�A[S�AZ  AY`BAW�AV-AU�
AU%AT �ASXAS"�AR�AR��AR�+AR�AQ�TAQ��AO�AH$�A9+A(Q�A#t�A�A�FAdZA-A	|�@��@ו�@�p�@���@��@��j@���@��@���@��@�r�@�;d@��@�%@��h@��@�JA���Aؙ�A�~�A�bNA�;dA��A�p�AָRA�dZA�"�A�hsA��A�S�A~��Ay�#Au�TAn�Al$�Aj^5Ah�AgVAfZAe�Aa�A^��A]��A\�/A\^5A\�A[��A[�AZ��AY�TAX(�AW\)AU�AU�hAU%ATZAT  AQ&�AN�uAKXAFjA;K�A5VA(�HA�A�A	��@�O�@�;d@�@�@��@��h@��T@�=q@��@��@�~�@�ff@�A�@���@�?}@���G�O�AڶFA�jA�33A��TAٍPA��yA�
=A֧�A�S�A� �A�v�A�|�A��;A��A�C�A��A|ȴAw`BAsdZAo�TAl�/Ak�Ai��Ag��Ad�!A`(�A^��A^�A]x�A\�\A[��A[7LAZ�HAZ9XAY�PAYXAYAXVAW�hAV��APAE7LA9�^A2$�A/�mA+�-A%G�A"Q�A��Ahs@��P@�"�@�E�@�@�Ĝ@���@�Z@�b@�n�@��9@�n�@�dZ@�A�@�Ĝ@�/@��7@���Aܥ�A܇+A�ffA�A�A���A�=qAڕ�A�l�A�&�A���A��wA���A�A��TA�ȴA�l�A�\)A��7A�&�A� �A�;dAy%AtE�ArffAn�DAi�Af(�Ad�/Ad^5Ab�A_"�A^5?A^�A]A]\)A]/A\�A[�mAZQ�AY%AL��AFA�A;t�A1��A*�+A"�\A�-A�;AȴA�A�9@߾w@��@�{@��@� �@���@��9@��@��
@��y@�o@��@��@�&�@���@��7A��A��yA�jA�5?Aء�A���A���AԁA�jAĴ9A�
=A��A�A�A���A�%A�7LA�z�A|jAw��As��Ao�;Am+Ai�^AfbAd��AdJAc/A`��A_`BA]�;A]O�A]"�A]�A]%A\�A\��A\�+A\1AZ�yAZ1'AVz�AM;dAAdZA,�\A&��A!"�A�wAȴA1A��@���@�x�@�Q�@�x�@��@���@�X@�@��7@�
=@�M�@�|�@��;@���@�G�@��^@��#A�S�A�p�AГuA�AΡ�A��A���A�-A�A�5?A��A�A�9XA�jA�^5A�$�A��\A�AydZAx�uAx(�Aw�AtE�Ar{Ap��AoC�AnffAk�;AjE�Ai��Ae��A_��A^ �A\r�A[�^A[x�A[?}A[VAZ�/AZ~�AP��AG��A@�yA3�TA,��A& �A�A�AA�A��A�w@�C�@�G�@��@�  @�-@�l�@�G�@� �@�ff@�;d@�33@��@���@�7L@�x�@��7A��/A��AҸRA��A��A�ȴA��
A�9XA��#A��`A�A��;A���A��;A�"�A�VA���A|9XA{�wAz�Ay�
AxffAwAvI�At��As;dArApv�Am�
Aj�DAf1'A_�^A]&�A\�DA\9XA\1A[��AZv�AY��AYXAQ�7AF��A:��A(�uA�A��A�DA+A+A	/A�@��y@�A�@���@���@�-@�+@�~�@�K�@���@�33@�  @��9@�V@�hs@���@���A��HA΋DA��A��TA�bNA��/A�A�A�r�A�VA� �A�;dA�%A�-A��A�7LA��PA��`A�ZA��A��A}��AxQ�As��Ar�!Ap-Anr�Amx�Al�Aj�Ait�Af��AedZAd�Ab�A`�A_S�A]S�A\��A\�\A\ffAT�9AIXA=x�A3A)��A"{A�hA��A"�A
�A bN@���@�@��@���@��@�&�@�{@�E�@�Q�@���@�r�@���@��G�O�G�O�G�O�A���A�1A��;A���A˼jA��A��`A�n�A��A��A��A��uA�ƨA�33A� �A�%A�S�A��DA�1A�S�A|v�Ay/Av�+AsS�An�uAk�TAj1Ai"�Ag�mAg33AfbNAe;dAd�Ab�AaXA^�DA\��A[�A[XA[;dAWC�AI�#A<ZA9��A.��A ��AffAC�A��A�@�5?@���@��@��@�I�@��
@�&�@�`B@�Z@�7L@��@�`B@��@��G�O�G�O�G�O�A�/A��A�=qA�7LA���A�O�A�I�A�  A��A���A�Q�A���A�&�A�G�A�
=A{&�Ay33AxE�AxbAu/Aqt�Ao��Al��Aj��Ai��Ah��Ah�Ag�mAf�Af�Ae�Ae�mAe�;Ae��Ae�Aap�A_��A^v�A]hsA[VAXȴAUƨAJ �A2�A'�A!�AA�A`BAJ@��w@�9@��@�bN@��@��R@��+@�-@�9X@�I�@�M�@���@�?}G�O�G�O�G�O�G�O�A؅A�A�n�AԲ-A�v�A�n�A���A���A�;dA� �A�/A�t�A�VA��-A���A�A�JA�+A��A���A}�^A{��A{t�A{AvA�Ap9XAkl�Ai%Ag�
AgdZAe�-Ad�Ad=qAb��AaO�A`n�A_��A^�\A]33A\VAZ��AR~�AA��A<�A6n�A*�A!S�A�A�TA
�`A%@�Z@���@�-@���@�%@��@�I�@�O�@��@��/@���@��G�O�G�O�G�O�G�O�A�7LAøRA�;dA�x�A��A���A��A��A�r�A��HA�oA���A�|�A�G�A}�A{"�Aw�;As�Am�-Al�jAk�Ai
=Af��Ad��AcAa��A`�RA^��A\�`A[dZAZĜAZ�DAZffAZQ�AZA�AZ5?AZ1AY�^AX�jAWAO�AI��AE|�A?%A1��A!K�A5?A33A
=AO�A�F@�9X@��m@���@�
=@�"�@��@�1@�b@�5?@� �@��`G�O�G�O�G�O�G�O�G�O�A�C�A�5?A��uA�E�A�bA��9A�G�A�A��-A��A�-A�G�A���A� �A�~�A� �A~��Ay��At��An��Aj��Ah�jAe�A^�DA["�AZ��AZbNAZM�AZI�AZI�AZA�AZ1'AZ�AY�AY�7AX��AWƨAU%AQp�AMp�AHĜAEƨA4JA ��A�HAZA�\A�7A\)A�!@�ff@�@� �@�dZ@�"�@�@���@��j@��R@�/@���@�dZG�O�G�O�G�O�G�O�G�O�A���A� �A��mA��A�n�A�&�A��`A�~�A��A��DA�A��A�O�A�S�A��^A�33A�ZA��A�dZA�FA}��A|bNAz~�Ax�`Aw�-At(�Aox�Aj��Ai�Ah��AhȴAh�RAh  Ae\)Abn�A`=qA^ZA\5?A[�PA[`BAS��AJZAH��ADI�A1�A#VA�TA(�A	;dAt�@���@�{@�A�@���@���@�p�@�9X@��@��h@�dZ@�?}@���@��G�O�G�O�G�O�G�O�APA�\)A��yA�bA���A���A�S�A�1'A�jA���A��wA�{A��#A�hsA��A��!A��7A�VA���A�%A��A��TA�ffAs�Al��Ad�`Ad �Ac��Acl�Ab{A_�A\��A\��A\E�A[��AZJAX5?AW��AWAU��AN�ALȴAEG�A21'A*��A?}AdZA�TA|�A~�@�bN@��@��y@�hs@��`@�x�@�E�@�K�@�{@�r�@��\@���G�O�G�O�G�O�G�O�G�O�A��A���A�M�A�M�A���A�VA��PA�bNA�ƨA�ĜA�-A���A�ZA�7LA��;A�bA���A���A���A��A�l�A�bA�^5A�jA���A���A}?}AyK�ArVAh�DAc�AbVAa��A`JA_7LA^ȴA]��A[�AYoAX1'AO�PAK�mA<r�A45?A-p�A��AhsA�\A�A1@��@�J@��@�ƨ@�Z@�@�O�@���@���@�  @�ff@�n�G�O�G�O�G�O�G�O�G�O�A��`A�A�A�~�A��A��A�|�A��A�$�A��A�A��A���A�ƨA��\A��RA���A�`BA���A���A� �A�A��A|��Azr�Ax-Av(�As�Am
=AiAf��AeVAb�A`jA\�jA[t�A[+AZ�AZz�AY��AV �AQp�AO�AM;dAAVA7hsA.��A#dZA5?A�A%A+@�`B@և+@̛�@� �@��y@���@�\)@��F@��@�b@���G�O�G�O�G�O�G�O�G�O�A�/A��AčPAÍPA��mA�JA�{A�VA�`BA��A�`BA�/A�  A�x�A� �A���A��7A��A���A�n�A��!A�Q�A���A�JA�t�Av�RAnbNAk�Ag?}Acl�A`-A^��A\��A[�^A[oAZ^5AZ-AY�AY�hAX�\ARz�AN�!AM��AC�A)p�AS�AoAG�A�;A��@���@�bN@�I�@��`@�\)@���@��h@�dZ@��@�Ĝ@��u@�{G�O�G�O�G�O�G�O�G�O�A��A���A�G�AƉ7A�t�A�hsA�^5A�K�A�7LA�{A���AļjA�`BA�Q�A��A���A���A���A���A�%A��mA��A�E�A{��AuhsAp  Ak��Ac�A`�+A`=qA_�A_�A_dZA_/A^��A]�;A]�hA]l�A]O�A\�`A[\)AT��AH$�A;+A3S�A+�A =qAȴA1A��@��-@�O�@ҸR@�?}@���@���@�V@���@��+@�r�@��7@��/G�O�G�O�G�O�G�O�G�O�A�S�A���A���A���Aʙ�A�jA�9XA���Aɗ�A�
=A�hsA���A�Q�A��A��A�+A���A�{A���A�x�A��A�VA���Az��Ay�PAwXAu�FAsl�Aq�;Ap=qAkAg��Af�9AeAe�Ab�yAap�A`~�A_��A^��AW|�ARJAK�7A?�FA4ffA+�mA��AoA�A��@�Z@ᙚ@��@��@���@��+@���@��@�O�@�j@�x�@���G�O�G�O�G�O�G�O�G�O�A�7LA�33A�&�A�A�oA���A��A��A���A͟�A�ZA�JAˁA�E�A��A�|�A���A���Aux�Aq��An��Al��Al�RAl �AjQ�Ah��Af�DAdAb��A`�uA^�jA];dA[�A["�AY�FAXȴAW��AW�AV�+AUG�AP��AM�PAG�A8ĜA5��A)G�A�9A�#A�AG�@��
@��T@��@��/@���@�"�@��@�Q�@�l�@��@�`B@�r�G�O�G�O�G�O�G�O�G�O�1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111 1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111 111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111   1111111111111111111111111111111111111111111111111111111111111111   111111111111111111111111111111111111111111111111111111111111111    111111111111111111111111111111111111111111111111111111111111111    11111111111111111111111111111111111111111111111111111111111111     11111111111111111111111111111111111111111111111111111111111111     111111111111111111111111111111111111111111111111111111111111111    11111111111111111111111111111111111111111111111111111111111111     11111111111111111111111111111111111111111111111111111111111111     11111111111111111111111111111111111111111111111111111111111111     11111111111111111111111111111111111111111111111111111111111111     11111111111111111111111111111111111111111111111111111111111111     11111111111111111111111111111111111111111111111111111111111111     11111111111111111111111111111111111111111111111111111111111111        ;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;oG�O�;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;oG�O�;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;oG�O�G�O�G�O�;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;oG�O�G�O�G�O�;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;oG�O�G�O�G�O�G�O�;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;oG�O�G�O�G�O�G�O�;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;oG�O�G�O�G�O�G�O�G�O�;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;oG�O�G�O�G�O�G�O�G�O�;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;oG�O�G�O�G�O�G�O�;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;oG�O�G�O�G�O�G�O�G�O�;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;oG�O�G�O�G�O�G�O�G�O�;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;oG�O�G�O�G�O�G�O�G�O�;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;oG�O�G�O�G�O�G�O�G�O�;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;oG�O�G�O�G�O�G�O�G�O�;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;oG�O�G�O�G�O�G�O�G�O�;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;oG�O�G�O�G�O�G�O�G�O�BBBVB�B,B?}BbNB��BDB#�BYB�B��B��Bz�B�)BĜBs�B!�B�B�B�+Bl�BC�BuB��B�ZB��B�wB��B��B�\B�B�B{�Bm�B[#BL�BK�BG�B-B��B�B�VBR�BhB��BaHB33B�B
�yB
k�B
M�B
�B	��B	��B
%B
	7B
VB
{B
D�B
�hB
�-B
�NB1BuB!�BE�BE�BF�BL�BW
BaHBv�B�LB(�BaHB�7B�DBȴB�-B{B|�B��B$�B��B��B\)B1'B�)B��B��B�PBs�BffBT�BD�B>wB:^B7LB-B!�BhB%B��B�B�B��B��Bk�B0!BhBB��B|�BZB;dB
�'B
I�B
+B
�B
B
B	��B
%B
\B
!�B
S�B
�B
��B
�fBBJG�O�B��B��BĜBɺB��B�#B�fB�B�#BÖB��B;dB�'B-BbB�B �B�bB=qB�B��B��B�\B�Bt�BiyBcTBT�BN�BL�BF�B?}B9XB7LB6FB49B2-B/B-B+B�B�5BgmB�B��B��B�BcTB=qB+B
z�B
R�B
49B
#�B
	7B
B
  B
B
1B
�B
&�B
�B
��B
�B%B{B$�B�B�B�B�B�B�B��B��BB�B��BB��B��Bx�BVB�B+B��B�TB�B��BƨB��B�hB�=B�B~�B|�Bz�Bw�Br�BjB^5BW
BM�BI�BE�B@�B>wB&�BhB��B��B}�BH�B�ZB�\BS�BDB
t�B
�B
�B
  B
B
B
B
1B
%B
oB
9XB
hsB
�B
�B
�BG�O�BBBBBB��B��B��B�B�TB�?B�7B,Bp�B%B�B��BiyBF�B'�BVB��B�B�HBÖB��B�uB�PB�1B�B{�Bv�Bs�Bn�BiyBgmBdZB^5BXBP�B�BÖBp�B9XB.BbB�/BƨB��BL�B
��B
F�B
;dB
JB	�B	��B
B
B

=B
�B
]/B
�PB
�3B
ɺB
�B+B�B+B6FBH�B`BB�JB�jB�B$�B1'B�B�hBP�B'�B�B��B��B�uB�+BB�B��BȴBp�BL�B:^B�B�B��BǮB��B�B�{B�VB�JB�=B�+B�B�By�Bm�BbNBB��Bx�B33B  BǮB��B�%Bo�BW
B
�yB
Q�B
oB	��B	�mB	��B
  B
+B
\B
{B
G�B
�B
��B
ĜB
��B�B�B�BB�HB�ZB�sB��BoB5?Bk�BɺB��B�!B�BS�BDBs�Bt�B�#B��Bk�BG�B&�BVB�B��BB�qB�3B��B�uB�7B�B�B�B�B�B�B~�By�Bq�Bl�BK�BB��BDB�ZB�jB��Bz�BF�B
�B
�3B
?}B
,B	�B	��B
  B
B

=B
oB
49B
VB
�B
��B
�/B
��B�B8RB>wB<jB8RB49B/B&�B!�B49B?}B^5B�BVBɺBe`B+B�hB'�BǮB�B}�Bw�Bo�BO�B<jB.B!�B�BB��B�B��B��B�7Bz�Bw�Bv�Bt�Bs�Bq�Bm�B!�B�)B��BA�BhB�;B�B�JBXB$�B
�B
�{B
9XB
�B	��B
B	��B
B
VB
�B
0!B
|�B
�FB
�HB
��B�B%�BhB�BN�B��B��B7LB�%B��BB�BoBVB.B� BaHBbB�qB�1B�%B~�Bu�BhsBaHBP�B>wB/B!�BhB��B�B�Bz�BffB`BB\)B[#BW
BN�BJ�BG�B	7B�LB^5B�
B�=Bv�Bl�B7LB&�B
�B
�jB
l�B
/B
oB	��B	�B	�B	�B	�B
B
)�B
p�B
��B
ŢB
�ZBB�BI�BG�BF�BM�BVBgmB~�B��BǮBB_;B�!B�wBaHB��B%�B�B�sB��B��B�hB]/B9XB/B�B1B��B�B�NB��B�FB�B��B�{B�Bw�Be`B_;B^5B\)B�B��BffB�B�)B��Bq�BL�B$�B
��B
��B
XB
33B
B	�B	�B	�yB	�B	��B
B
;dB
aHB
�B
��G�O�G�O�G�O�B�B�B�B)�B@�By�B�5B<jB�B\)B#�B!�BVB��B��B��B�DBiyB��BĜB�=BjBP�B2-B%B�B�/B��BȴBB�XB�B��B�bB�Bn�B_;BVBS�BR�B1'BƨBiyBVB��B��BhsB2-B
�yB
��B
�VB
C�B
�B
B	��B	�sB	�`B	�B	��B	��B
1'B
ffB
~�B
�=G�O�G�O�G�O�B�B!�BH�Bp�B�uB��B�B��B��B�BiyBv�B7LB�%B�XB~�Bm�BffBcTBH�B&�B{B��B�sB�/B�B��B��B��B�^B�RB�RB�LB�FB�B�DBy�Bp�BffBR�B>wB&�B��B�B��B��B�DBk�B2-BbB
�B
aHB
{B
B	��B	�B	�sB	�sB	�B
JB
bNB
�+B
��G�O�G�O�G�O�G�O�BK�BH�BE�B�=B��B�wB��B��B�B��B��BPBE�BiyBO�B:^B-B�B1B�NB�uB�B� By�BK�B�B�B�
B��BƨB�LB�B��B��B�\B�+B� Bu�BiyBaHBO�BVB�bBm�B9XB�NB��Bo�B8RB
��B
��B
�+B
bB
+B	�B	�B	�TB	�BB	��B
�B
D�B
�+B
��G�O�G�O�G�O�G�O�B�VB�7Bx�BZB?}B�B  B��B��B�B{�BQ�BJ�B6FB�BB�BB�3B�+B~�Br�B\)BF�B6FB$�B�B\B��B�B�;B�B�B�B��B��B��B��B��BĜB�RB|�BQ�B1'B��B�hB"�B\B%B
ɺB
�B
iyB

=B	�LB	��B	�DB	y�B	l�B	r�B	x�B	�oB	�NB
�G�O�G�O�G�O�G�O�G�O�B8RB7LB6FB7LB6FB7LB6FB7LB9XBuB�yB?}B=qB<jB#�BVB�BƨB��Bl�BN�B=qB"�B�B��B��BȴBǮBǮBǮBƨBƨBŢBB�wB�FB�B��B}�BaHB>wB �B�{BoBBB
�fB
��B
�B
F�B
�B	��B	��B	�+B	r�B	ffB	cTB	jB	� B	�1B	�FB	�TG�O�G�O�G�O�G�O�G�O�B{BhBPB
=B%BB��B�B�`B�#B��B�3B  B33B^5B]/B	7B�B�\B�JB{�Bl�B\)BO�BC�B#�B��B�BɺBƨBƨBŢB�wB��B�DBu�BdZBP�BJ�BG�B+B��B�FB�DBB��BbNB9XB
�5B
��B
x�B
9XB
�B
	7B	�B	�5B	�B	�B	�B
B
�B
T�B
`BG�O�G�O�G�O�G�O�B��B��B��B��B��B��B�3B�qBɺB�mB%B�B��B�BdZBw�B�-BÖBz�B@�B�B�#B�%B�B�TB��B��B��B�{B�1Bl�BZBW
BS�BL�B=qB-B&�B!�B�B�NB��B��B	7B��B}�BS�B'�B
�B
�-B
r�B
I�B
6FB
�B
	7B	�B	�B	�BB	��B

=B
"�B
k�G�O�G�O�G�O�G�O�G�O�B�NB�HB�HB�B  BuB<jBs�BaHBp�B��B�;B��B�sBƨBs�BF�B9XB�B��B�B�B�B�B�9B��BjBK�BhB�}B��B�DB�%Bv�Bn�BjB`BBH�B6FB.B�sB��BQ�B�B�fBw�B@�B!�B!�BB
_;B
>wB
'�B
!�B
\B	�B	�HB	�B	�B
%B
�B
,G�O�G�O�G�O�G�O�G�O�BǮBƨB�wB��Bo�Be`Be`B}�B��B��B%BB�B�VB�B-BbBJB�mBq�BǮB�{Bz�B\)BK�B>wB/B�B�BB��B�!B��B�+Bw�BYBL�BI�BF�BB�B:^B�B��B�yB��Bt�B0!B�B��BH�BVB
��B
��B
[#B
�B
VB	��B	�fB	��B	�B	�sB	��B
�B
D�G�O�G�O�G�O�G�O�G�O�B��B��B��B��B��B��B�B�^BƨB��B��B�
B��BĜBÖB�BB'�B�FB+B�XB?}B��B�B�^B1'B�B��B�!B�VBs�BgmBXBN�BG�BA�B?}B=qB9XB/B��B�;B�
B�+B��Bl�B2-B%B
��B
��B
�%B
'�B
�B
1B	��B	�sB	�)B	�B	��B
PB
;dB
aHG�O�G�O�G�O�G�O�G�O�B�uB�oB�hB�oB�{B��B��B��B��B��B��B��BaHBaHB�B:^B��Bm�B�BcTBjB �B�XBjB1'BB�B�DBv�Bs�Bo�Bm�Bl�BjBhsB`BB]/B\)B[#BXBI�BhB�BL�B�B�B�%B]/B9XBuB
k�B
0!B
"�B
B	��B	�HB	�)B	�B	��B
	7B
#�B
F�G�O�G�O�G�O�G�O�G�O�B�TB�HB�BB�;B�5B�/B�)B�B�B��B��B�9B��B�;BD�B�Bp�B�B��B��BW
BɺB��BiyB[#BG�B8RB#�B�BB�B�XB�B��B��B�PB� Bw�Bo�BffB'�B��BƨBl�B�B�/B�Bk�B�B
��B
�B
D�B
�B
%B	��B	�yB	�B	�;B	��B
B
bB
'�G�O�G�O�G�O�G�O�G�O�B�B�B�B�B�B�B�B��BDB�B<jB}�B��B��Bp�BBffBB=qB�B��B�B�fB�BB��B��B�B�{B�1Bu�Be`BYBM�BE�B9XB1'B&�B#�B�B�B�B�
B��B=qB%�BŢBVB�B  B
�B
m�B
6FB
B	��B	�B	�;B	��B	��B	�ZB	��B
VB
6FG�O�G�O�G�O�G�O�G�O�1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111 1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111 111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111   1111111111111111111111111111111111111111111111111111111111111111   111111111111111111111111111111111111111111111111111111111111111    111111111111111111111111111111111111111111111111111111111111111    44444444444444444444444444444444444444444444444444444444444444     44444444444444444444444444444444444444444444444444444444444444     111111111111111111111111111111111111111111111111111111111111111    11111111111111111111111111111111111111111111111111111111111111     11111111111111111111111111111111111111111111111111111111111111     11111111111111111111111111111111111111111111111111111111111111     11111111111111111111111111111111111111111111111111111111111111     22222222222222222222222222222222222222222222222222222222222222     22222222222222222222222222222222222222222222222222222222222222     22222222222222222222222222222222222222222222222222222222222222        B
=BJB�B �B49BH�Bm�B�'BuB.Be`B�{B�B�3B��B{B�NB�PB1'B%B�dB�oBx�BQ�B�BB�B�#BɺB�B��B��B�JB�7B�Bw�Be`BS�BS�BO�B6FBB�FB��B]/B�B�)BjB;dB&�B
�B
s�B
VB
�B
B
B
PB
bB
�B
�B
K�B
��B
�XB
�yB\B�B(�BO�BO�BP�BW
BaHBl�B�B��B>wBs�B��B��B�`B��B5?B��B�B<jB�)B�RBk�BE�B�B�B��B��B~�Br�BaHBN�BH�BD�BB�B8RB.B�BhBB��B��B��B�3Bv�B:^B�BbB�B�1BdZBF�B
�jB
S�B
5?B
 �B
DB
VB
1B
\B
�B
+B
]/B
�PB
�3B
�BDB�G�O�B��B��B��B�B�5B�sB��BB��B�B��BgmB��B=qB$�B��BF�B�BT�BB��B��B��B�bB�Bu�Bq�BbNB[#BYBR�BK�BE�BC�BB�B@�B>wB;dB9XB7LB#�B�Bw�B��B�#B�B�hBp�BJ�B�B
�+B
_;B
@�B
0!B
�B
bB
DB
PB
uB
 �B
2-B
�JB
�
B
��BhB�B0!B  B��B��B��BBBBVB49B#�B��B�B�B�'B�DBk�B0!B�B+B�B�sB�NB�B�dB��B��B�hB�PB�DB�7B�%B�By�Bm�BffB\)BXBS�BN�BL�B5?B�B+B�NB�PBZB��B��BdZB�B
�B
,B
'�B
VB
\B
bB
hB
�B
uB
 �B
F�B
u�B
�^B
�fB
��BhG�O�B�B�B{B{BuBhBbB'�BBbB�B�LB\)B�hB�B�B�B}�B[#B;dB �BhBB��B�B�-B��B��B��B�oB�PB�+B�B� By�Bw�Bu�Bo�BiyBcTB-B�
B�BI�B?}B!�B�B�B�-B_;B
�RB
W
B
K�B
�B	��B
JB
oB
�B
�B
+B
l�B
��B
B
�B
��B�B+B>wBI�B\)Bs�B��B��BBC�BL�B\)B�dBq�BG�BbB�5B�B��B��BbNB�B�mB�7BaHBP�B/B1B�mB�#B�B��B��B��B��B��B��B��B�{B�VB�Bw�B�B�HB�VBG�B{B�#B�LB��B�BjB
��B
e`B
$�B
PB	��B
bB
oB
�B
!�B
&�B
YB
��B
�LB
�B%B+B1'B��B��B��B��BoB)�BN�B�=B�B'�B��B��Bt�B5?B��B��B��B�!B�BaHB>wB&�B+B�`B�B��B��B�?B��B��B��B��B��B��B��B��B�{B�\B�+B�BcTB�B�dB!�B��B��B�RB�hB^5B%B
ɺB
S�B
A�B
%B
VB
{B
�B
�B
&�B
H�B
iyB
��B
�^B
�B\B49BL�BXBVBQ�BM�BI�BB�B@�BP�B`BB��B�B0!B�B�BT�B�LBK�B�sB��B��B�\B�=BiyBT�BF�B9XB33B�BVB
=B�sB�!B��B�oB�\B�VB�JB�DB�7B�+B;dB��BBZB)�B��BÖB��Bp�B<jB	7B
�B
P�B
33B
�B
�B
bB
�B
$�B
,B
F�B
�uB
��B
��BJB-B<jB-B9XBk�B�wB�BW
B��B�}B�ZBoB@�B��BcTB��B�B49B�5B��B��B��B�bB�B{�Bl�BYBI�B<jB-B�B��B��B��B� By�Bu�Bt�Bq�BhsBdZBcTB$�B��B{�B�B��B�\B�+BP�BA�B%B
�B
�%B
H�B
+B
{B
	7B
+B
B

=B
�B
B�B
�7B
�9B
�5B
��B�B/BffBdZBe`BjBr�B�B��B�^B�fB!�B�BB�B��B��BI�B\B+B�BĜB�-B}�BVBL�B6FB$�B�BDB��B�B��BǮB�jB�-B��B��B�Bz�By�By�B>wB�;B�B<jB��B��B�VBiyBA�BoB
ƨB
s�B
N�B
�B
VB
JB
B
B
bB
�B
VB
{�B
��B
�}G�O�G�O�G�O�B<jB;dB<jBJ�BbNB��BBu�B��B�+BC�BC�B6FB��BǮB�FB�B��B%�B�yB�B�DBr�BT�B&�BPB��B�B�mB�HB�B��B��B�B��B�VB}�Bs�Bq�Bq�BQ�B�mB�+Bv�B�B�LB�+BQ�B+B
�yB
�B
aHB
8RB
�B
�B
B
B
DB
�B
�B
M�B
�B
��B
��G�O�G�O�G�O�B>wBC�Bk�B�uB�FB��B��B�B�BK�B��BBn�BB�NB��B�VB�+B�%Bl�BH�B7LB�B	7B��B��B�B�B�NB�B�B�B�
B�
B��B�B��B�hB�1Bs�B_;BI�B��BB�B�B��B�B�PBR�B1'B
��B
�B
49B
$�B
uB
uB
1B
+B
oB
+B
�B
��B
ÖG�O�G�O�G�O�G�O�Bn�Bk�Bl�B�RBƨB�sB%�B�mB��BǮB��B>wBq�B��Bw�BaHBS�B>wB/BVB�RB��B��B��Bt�B>wBoB��B�B�B�B��B��B��B�-B��B��B��B�JB�Bs�B5?B�3B�hB^5B%BÖB�uB\)B�B
�NB
��B
2-B
(�B
uB
hB
B
B
�B
=qB
e`B
��B
ÖG�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�B?}B=qB7LB49B0!B+B$�B�B\BB��B�B49BjB��B��B@�B�B�^B�LB��B��B�+By�Bp�BQ�B+BB�B�B�B�B�B��B�FB��B�\By�Bs�Br�B2-B�B�BB�RB.BÖB�JBe`B1B
��B
��B
aHB
=qB
1'B
�B
%B
B
B
�B
-B
>wB
{�B
�1G�O�G�O�G�O�G�O�B��B��B��B��B��B��B�;B�B��B�B5?BQ�B)�B�NB�3B��B�HB��B�Bq�BQ�BoB�}BP�B�B��BǮBŢB��B�FB��B�B�B� Bx�BjBXBQ�BM�BB�BPBBŢB6FBB��B� BS�B�B
�/B
��B
s�B
`BB
E�B
33B
{B
B

=B
&�B
49B
L�B
��G�O�G�O�G�O�G�O�G�O�BbBbBhB�B/BB�Bn�B��B�uB��B�;BVB-B!�B
=B�Bu�BjBT�B&�B�B�B�BbB�`B�#B��B�BI�B�BȴB�XB�?B��B��B��B�bBw�BdZB]/B�B��B�BH�B�B��Bn�BM�BN�B0!B
�JB
jB
S�B
M�B
<jB
�B
PB
B
�B
2-B
C�B
XG�O�G�O�G�O�G�O�G�O�B��B��B�B�)B��B��B��B�?B�)B+B8RBs�B�}BO�BffBA�B@�B#�B�FB��BƨB�B�VB}�Bp�BaHBP�B{B�B�HB��B�RB�B�=B|�Bx�Bv�Br�Bm�BM�B(�B�B%B��BaHB �B��By�B=qB(�B
��B
�=B
G�B
=qB
%�B
{B
B
%B
�B
(�B
J�B
r�G�O�G�O�G�O�G�O�G�O�B��B��B��B��B��B�B�`B�B��BB+BbB	7B��B��BW
B=qBZB�yBI�B��B�%B�B\)B��Bl�B�B%B�fBÖB��B��B�DB�By�Bs�Bq�Bo�Bk�BbNB1'BbBDB�}B��B��BdZB7LB.B.B
�RB
YB
E�B
8RB
'�B
�B
JB

=B
&�B
=qB
k�B
�hG�O�G�O�G�O�G�O�G�O�BȴBȴBǮBǮBɺB��B��B��B��B�B�B�#B��B��B��Bq�B��B�3Bl�B�FB��BaHB��B��Bk�B:^B�BB�B��B��B��B��B��B��B�{B�hB�bB�\B�JB~�BH�B�TB�BJ�BbB�^B�hBm�BH�B
��B
bNB
VB
6FB
(�B
uB
\B
"�B
0!B
;dB
VB
x�G�O�G�O�G�O�G�O�G�O�B�B�B�B�B�B{BuBoBbBDB+B��B�NB'�B� B
=B�B%B�yBE�B��B%B�#B��B�uB� Bq�B\)BM�B?}B{B�B�fB�5B�BĜB�LB�B��B��B_;B2-B  B��BS�B�B�dB��BM�B49B
�dB
y�B
M�B
:^B
33B
�B
\B
{B
,B
7LB
D�B
\)G�O�G�O�G�O�G�O�G�O�B(�B(�B(�B(�B$�B&�B)�B33BE�BT�Bv�B�jBE�BVB�BR�B��BR�B{�BR�B8RB#�B�B�B	7B��B�fB��BB�!B��B�oB�+B~�Br�BjB_;B\)BXBN�B,BbB�NBu�BaHBB�bBP�B7LB'�B
��B
n�B
:^B
,B
#�B
�B
	7B
DB
�B
/B
D�B
l�G�O�G�O�G�O�G�O�G�O�1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111 1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111 111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111   1111111111111111111111111111111111111111111111111111111111111111   111111111111111111111111111111111111111111111111111111111111111    111111111111111111111111111111111111111111111111111111111111111    44444444444444444444444444444444444444444444444444444444444444     44444444444444444444444444444444444444444444444444444444444444     111111111111111111111111111111111111111111111111111111111111111    11111111111111111111111111111111111111111111111111111111111111     11111111111111111111111111111111111111111111111111111111111111     11111111111111111111111111111111111111111111111111111111111111     11111111111111111111111111111111111111111111111111111111111111     22222222222222222222222222222222222222222222222222222222222222     22222222222222222222222222222222222222222222222222222222222222     22222222222222222222222222222222222222222222222222222222222222        <�t�<���<�t�<���<���<�1{<�|<��c<���<��@<�U�<���<���<�~<��Z=Q��<�%<�|�<��'<���<��,<�=�<�o�<��<�$t<��@<��O<��@<���<��<���<�ȟ<���<���<�1{<�-#<��H<�t�<���<���<�1{<�2<���<�[l<��@<�2<�Nf<�Ft<���<�m�<�i/<���<��q<���<�m�<�X�<�C�<�C�<�C�<�X�<�C�<�C�<�C�<�C�<�C�<�C�<�C�<�X�<�C�<�X�<�X�<�X�<��j<���<��	<��\<���<��|<�s.<�,�<�AJ<�}�<�h=K4<�W <�J�<�IR<�O"<��<���<�?><��x<��<�?><��!<��<���<�X�<�C�<��q<��b<��<��&<� T<�{J<�'g<�{J<�fQ<�M<�a�<�{J<�'g<�8	<�H�<��<<�fQ<�8	<���<�QY<�'g<�<`<�o<�'g<�o<�o<�o<�o<�o<�o<�o<�o<�o<�oG�O�<�QY<�o<�o<�'g<�<`<��<<�	�<��<���<�*E=�=�<��<��.<�pe<�$5<�ϖ<�1<�J<�]�<��<w<w<<we�<v�)<vr<y	l<vjU<u�<v@d<v@d<v@d<u<u<u<u<u�<u<u<vr<w��<~q�<��;<w<w<<vjU<v�F<w<w��<x��<v�F<v@d<u�<v@d<u�<u�<u<u<u<u<u<u<u<u<u<u<u<u�<u<u<u<u�<v@d<v�8<�=v�=.	�=F�l=	�/<�
R<y	l<oܜ<���<k��<g�u<h�<f[�<e�$<f��<l��<l�<e�<f<e�3<e`B<e`B<e�3<e�3<f1�<gW�<e�<fپ<e�3<e�3<e�$<e�3<e�$<e�$<e�<f[�<j�R<gW�<k�	<l�<i$�<j �<hS;<g�g<e�$<e�<e�3<e�3<e`B<e`B<e`B<e`B<e`B<e`B<e`B<e`B<e`B<e`B<e`BG�O�<U'�<e�3<e�3<UQ�<Uϫ<V�b<Y�=�<r'�<���<�Pr=O�=
\�<�&�<n=<m3<i��<`��<^*�<[ߏ<Ws<V�S<V�E<])<c��<V�S<U{�<UQ�<U��<UQ�<U{�<U'�<UQ�<UQ�<T��<U'�<UQ�<U{�<U��<W6<Z<!<[a�<W��<UQ�<Uϫ<V�E<U{�<V#�<Y�><X�<Ws<U{�<U��<U��<T��<T��<T��<U'�<U'�<T��<T��<T��<T��<T��<T��<T��<T��<D��<D��<D��<Ez<Em]<E�N<G��<�a<m�K=KhI<�g�<���<�O�<�@�<���<I[�<F��<���<�?<^�z<���<Y@y<G�<P=�<T�<P=�<F#<D�<H6e<K�3<Em]<D��<Dŗ<Dŗ<D��<Dŗ<Ez<F��<F#<K�$<G:�<J��<I��<G��<Ht<E�1<Em]<Ez<Ez<F��<H�H<F?<Ez<Em]<D��<D��<D��<D��<D��<D��<D��<D��<D��<D��<D��<D��<D��<4cI<4�<4cI<5<55<7�<AT�<{*�<��b<�	l<��b<])<{��<�#�=	V�<Կ
<mgw<G�<BPr<AT�<:�<>�<?�[<60�<4�;<4�<9�Z<5��<60�<4�;<49X<49X<49X<49X<49X<49X<4cI<5^�<4�,<5<8��<;�<K�$<6�o<60�<4�<5��<7`<<�S<5<7�&<4�<5��<4�;<4cI<49X<49X<49X<4cI<49X<49X<49X<49X<49X<49X<49X<49X<60�<60�<6�<7�4<;n�<?�j<T��<C��<e�$=�<��<w<<��<P�<���<�28<~�<a�<%�j<$ �<$T�<,��<(�U<%�M<%�[<$��<)��<&v!<$T�</��<F#<'�<&�<$T�<#�
<#�
<#�
<#�
<$ �<)i<)8<&�<.Se<'��<&�<'�<$��<&��<'q�<$~�<%zx<'�<$~�<$T�<$*�<$ �<$ �<$ �<$ �<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<%�M<)��<--�</��<.�9<;D�<$ҳ<3�<L��<LN�<�g�=ZQ<�l�<0��<d�m<iN�<L��<$~�<$��<$ҳ<%�[<$T�<%�[<&�<%�[<%P�<&L0<*�<-Ց<4�<G��<+�N<$T�<$ �<#�
<$ �<%&�<$~�<$ �<'q�<*�<,�<7�	<+�N<$T�<$ �<'�<$*�<'Ŭ<$T�<%�[<%P�<$T�<$*�<$ �<$*�<$T�<$ �<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<$*�<$~�<(mr<$��<$ҳ<$��<%�[<'q�<(C�<.�H<P�}=\�<�n/<��L<�O�<UQ�<&"><+�@<2��<(C�<;�<6Z�<%�j<)i<&�<$ҳ<%�j<%&�<%�M<+�<%zx<%zx<%&�<'��<&L0<'��<$~�<#�
<#�
<&��<+�N<,�<*�<)i<'��<&�<%P�<%�M<&�<%&�<&L0<$T�<$��<$*�<$ �<$T�<$ �<#�
<$ �<#�
<#�
<#�
<#�
<#�
G�O�G�O�G�O�<%�j<'�<'G�<'q�<.�9<;n�<Cv<�5�=*y�<���<%zx<-Ց<`L<C��<)�<%�[<.�+<�[l<U{�<Dŗ</%<*��<-W�<7�&<+�]<'G�<$��<%P�<$T�<$~�<%&�<%&�<&L0<%P�<*�<&��<%zx<#�
<#�
<$��<-��<.�+<$~�<*�</��<&�<'G�<+6z<$��<$��<&"><$T�<$T�<$ �<$T�<$ �<$ �<$ �<#�
<#�
<#�
<#�
<#�
<#�
G�O�G�O�G�O�<#�
<&v!<'�<)��<)�</%<2��<0J�<=�
<���<ɯ�=/�W<�SP<�
<W�
<)8<$ҳ<#�
<*��<0t�<'G�<*:�<'Ŭ<%P�<$��<$ �<$T�<$ҳ<$~�<#�
<#�
<#�
<#�
<$T�</%<'q�<%&�<$��<(�F<$*�<$T�<+`k<B&�<,��<&L0<$~�<$��<'G�<%�[<&"><%�M<&"><$*�<$ �<$ �<$ �<$*�<$ �<$ �<#�
<#�
<#�
<#�
G�O�G�O�G�O�G�O�<$~�<$��<7�4<y�#<M�g<M�g<L��<,�<'�<<�=c�^<��U<]/<��<=<6<2B<0��<.Se<0��<d�m<(mr<$ �<$ �<7�4<C")<9�w<)��<%P�<$*�<&v!<$��<$ �<%P�<&�<$��<$��<$ҳ<%�j<$��<$ �<'��<3��<%�j<&L0<,1<)�<&�<'G�<)?)<$��<$��<&v!<%&�<$T�<$ �<$ �<$*�<$*�<#�
<#�
<#�
<#�
<#�
G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�<%P�<(mr<$*�<$ �<$*�<$*�<$~�<$ҳ<$~�<$ҳ<?3�<��[<j�o<��<�D(<�uy<�	<<�S<%�j<&�<&L0<'G�<&v!<%P�<.�9<7,R<7�4<&��<$ �<#�
<#�
<$T�<)��<,1<(�c<'q�<(C�<$T�<#�
<'q�<)8<$ �<%&�<6��<2<'G�<&"><.}V<(mr<$~�<%&�<$~�<$ �<$T�<$T�<$*�<$ �<$ �<$ �<#�
<#�
<#�
<#�
G�O�G�O�G�O�G�O�<$ �<)��<%�M<$*�<$ �<$ �<#�
<%�j<\]d<M��<,��<>�<ECl<p�b=��<.)t<+`k<Xn�<<�b<:I=<c<z<���<J-�<X�<%P�<#�
<$*�<%zx<,1<)?)<$ �<$ �<$T�<%�M<'�<$T�<$*�<%�j<&�<$*�<'�<8��<(�<,2#<&L0<%�[<(�c<)?)<$��<$~�<$ �<$*�<$ �<$~�<$��<$*�<$*�<#�
<#�
<#�
<#�
G�O�G�O�G�O�G�O�G�O�<4cI<5��<7`<5<4�<60�<A �<c>�<A~�<;¹<��O<4�<@/0<y3]<�7"<��<5��<:K<j �<>�<4�<5��<@><^�z<>7�<hS;<A~�<XD�<w��<Fi<7`<4�;<7`<4�<4cI<5<:I=<8'�<5<8Q�<5<@Y!<(C�<&�<1m<'��<%�j<#�
<$ҳ<*��<$T�<$ �<#�
<$ �<$ҳ<$*�<$T�<$~�<#�
<#�
<#�
<#�
G�O�G�O�G�O�G�O�G�O�<KSP<F?<Rܱ<���<�$<[a�<a�<e�$<d�<[7�<H�H<D�<Dŗ<zX�<nc <Ez<L%<�i�<��<d:�<8{�<<j<9M�<8��<7�<8{�<X�<CL<8Q�<7�4<;��<7`<?�j<6�<4cI<49X<4cI<4�<=��<5��<4cI<4�;<<@�<9�h<8��<;n�<;¹<8{�<4�<5��<5^�<5��<4�;<4�,<4cI<4�,<4�;<4cI<49X<49X<49X<49XG�O�G�O�G�O�G�O�G�O�<T��<U��<WI(<U��<V�b<W6<[ߏ<]��<be<d�<`L<kC<c��<bCW<\�U<u<q6<UQ�<U'�<� �<���<��<�c^<~�m<�#�<z/<MJ�<P�<P�<M �<F��<G��<E�N<D�<Ez<D��<D��<Dŗ<Em]<Fi<Em]<D��<I[�<d:�<Mt�<H�H<F��<Dŗ<D��<G��<G�<Dŗ<Dŗ<Dŗ<Dŗ<D�<D�<D�<D��<D��<D��<D��G�O�G�O�G�O�G�O�G�O�<T��<U��<VM<T��<T��<T��<T��<T��<T��<T��<X�<��<l��<�;�<���<V�S<e`B<�d<�~�=�<�+�<�}k<��<q�
<j �<a�t<�{J<[��<U'�<UQ�<T��<T��<T��<T��<Uϫ<U'�<T��<T��<U'�<U'�<V�S<\3r<])<XD�<X�<[7�<V�S<V#�<Vwp<Z�<V#�<U'�<U{�<U'�<U{�<U'�<U��<T��<T��<T��<T��<T��G�O�G�O�G�O�G�O�G�O�<e�<f��<e�3<e`B<e`B<e`B<e�3<e�$<f<f1�<iN�<��M<�$_<���<nc <z��<|&W=_�<�J�<��o<�I�<zX�<~q�<g�<h�<gW�<h�<g-�<g-�<r{�<p�<f��<f<e�$<h},<g�<f<e�<f<g��<f��<g-�<kp&<kF5<h�<k�	<f<j�R<f��<g�g<f��<f<e�3<e�3<e�$<e�$<e�$<e�$<e`B<e`B<e`B<e`BG�O�G�O�G�O�G�O�G�O�<u<u<vr<v�F<u<u<u<u<u<�'g<�'g<���<��\=[��=8�K<���<�+<�s<�fQ<zX�<xa�<u�<u�<w��<w<<x�z<y�#<v�)<x��<x�<w<<v�)<v@d<w<vjU<v�8<u�<u�<v�8<v�F<v@d<w<<~�|<vjU<{�t<~�m<y�1<vjU<vr<x��<vjU<v�F<u�<u�<u�<u�<u�<u<u<u<u<uG�O�G�O�G�O�G�O�G�O�PRES            TEMP            PSAL            PRES            TEMP            PSAL            PRES            TEMP            PSAL            PRES            TEMP            PSAL            PRES            TEMP            PSAL            PRES            TEMP            PSAL            PRES            TEMP            PSAL            PRES            TEMP            PSAL            PRES            TEMP            PSAL            PRES            TEMP            PSAL            PRES            TEMP            PSAL            PRES            TEMP            PSAL            PRES            TEMP            PSAL            PRES            TEMP            PSAL            PRES            TEMP            PSAL            PRES            TEMP            PSAL            PRES            TEMP            PSAL            PRES            TEMP            PSAL            PRES            TEMP            PSAL            PRES            TEMP            PSAL            PRES            TEMP            PSAL            PRES            TEMP            PSAL            PRES            TEMP            PSAL            PRES_ADJUSTED = PRES                                                                                                                                                                                                                                            TEMP_ADJUSTED = TEMP                                                                                                                                                                                                                                            PSAL_ADJ = CTM_ADJ_PSAL + dS, dS is calculated from a potential conductivity (ref to 0 dbar) multiplicative adjustment term r.                                                                                                                                  PRES_ADJUSTED = PRES                                                                                                                                                                                                                                            TEMP_ADJUSTED = TEMP                                                                                                                                                                                                                                            PSAL_ADJ = CTM_ADJ_PSAL + dS, dS is calculated from a potential conductivity (ref to 0 dbar) multiplicative adjustment term r.                                                                                                                                  PRES_ADJUSTED = PRES                                                                                                                                                                                                                                            TEMP_ADJUSTED = TEMP                                                                                                                                                                                                                                            PSAL_ADJ = CTM_ADJ_PSAL + dS, dS is calculated from a potential conductivity (ref to 0 dbar) multiplicative adjustment term r.                                                                                                                                  PRES_ADJUSTED = PRES                                                                                                                                                                                                                                            TEMP_ADJUSTED = TEMP                                                                                                                                                                                                                                            PSAL_ADJ = CTM_ADJ_PSAL + dS, dS is calculated from a potential conductivity (ref to 0 dbar) multiplicative adjustment term r.                                                                                                                                  PRES_ADJUSTED = PRES                                                                                                                                                                                                                                            TEMP_ADJUSTED = TEMP                                                                                                                                                                                                                                            PSAL_ADJ = CTM_ADJ_PSAL + dS, dS is calculated from a potential conductivity (ref to 0 dbar) multiplicative adjustment term r.                                                                                                                                  PRES_ADJUSTED = PRES                                                                                                                                                                                                                                            TEMP_ADJUSTED = TEMP                                                                                                                                                                                                                                            PSAL_ADJ = CTM_ADJ_PSAL + dS, dS is calculated from a potential conductivity (ref to 0 dbar) multiplicative adjustment term r.                                                                                                                                  PRES_ADJUSTED = PRES                                                                                                                                                                                                                                            TEMP_ADJUSTED = TEMP                                                                                                                                                                                                                                            PSAL_ADJ = CTM_ADJ_PSAL + dS, dS is calculated from a potential conductivity (ref to 0 dbar) multiplicative adjustment term r.                                                                                                                                  PRES_ADJUSTED = PRES                                                                                                                                                                                                                                            TEMP_ADJUSTED = TEMP                                                                                                                                                                                                                                            PSAL_ADJ = CTM_ADJ_PSAL + dS, dS is calculated from a potential conductivity (ref to 0 dbar) multiplicative adjustment term r.                                                                                                                                  PRES_ADJUSTED = PRES                                                                                                                                                                                                                                            TEMP_ADJUSTED = TEMP                                                                                                                                                                                                                                            PSAL_ADJ = CTM_ADJ_PSAL + dS, dS is calculated from a potential conductivity (ref to 0 dbar) multiplicative adjustment term r.                                                                                                                                  PRES_ADJUSTED = PRES                                                                                                                                                                                                                                            TEMP_ADJUSTED = TEMP                                                                                                                                                                                                                                            PSAL_ADJ = CTM_ADJ_PSAL + dS, dS is calculated from a potential conductivity (ref to 0 dbar) multiplicative adjustment term r.                                                                                                                                  PRES_ADJUSTED = PRES                                                                                                                                                                                                                                            TEMP_ADJUSTED = TEMP                                                                                                                                                                                                                                            PSAL_ADJ = CTM_ADJ_PSAL + dS, dS is calculated from a potential conductivity (ref to 0 dbar) multiplicative adjustment term r.                                                                                                                                  PRES_ADJUSTED = PRES                                                                                                                                                                                                                                            TEMP_ADJUSTED = TEMP                                                                                                                                                                                                                                            PSAL_ADJ = CTM_ADJ_PSAL + dS, dS is calculated from a potential conductivity (ref to 0 dbar) multiplicative adjustment term r.                                                                                                                                  PRES_ADJUSTED = PRES                                                                                                                                                                                                                                            TEMP_ADJUSTED = TEMP                                                                                                                                                                                                                                            PSAL_ADJ = CTM_ADJ_PSAL + dS, dS is calculated from a potential conductivity (ref to 0 dbar) multiplicative adjustment term r.                                                                                                                                  PRES_ADJUSTED = PRES                                                                                                                                                                                                                                            TEMP_ADJUSTED = TEMP                                                                                                                                                                                                                                            None + dS, dS is calculated from a potential conductivity (ref to 0 dbar) multiplicative adjustment term r.                                                                                                                                                     PRES_ADJUSTED = PRES                                                                                                                                                                                                                                            TEMP_ADJUSTED = TEMP                                                                                                                                                                                                                                            None + dS, dS is calculated from a potential conductivity (ref to 0 dbar) multiplicative adjustment term r.                                                                                                                                                     PRES_ADJUSTED = PRES                                                                                                                                                                                                                                            TEMP_ADJUSTED = TEMP                                                                                                                                                                                                                                            PSAL_ADJ = CTM_ADJ_PSAL + dS, dS is calculated from a potential conductivity (ref to 0 dbar) multiplicative adjustment term r.                                                                                                                                  PRES_ADJUSTED = PRES                                                                                                                                                                                                                                            TEMP_ADJUSTED = TEMP                                                                                                                                                                                                                                            PSAL_ADJ = CTM_ADJ_PSAL + dS, dS is calculated from a potential conductivity (ref to 0 dbar) multiplicative adjustment term r.                                                                                                                                  PRES_ADJUSTED = PRES                                                                                                                                                                                                                                            TEMP_ADJUSTED = TEMP                                                                                                                                                                                                                                            PSAL_ADJ = CTM_ADJ_PSAL + dS, dS is calculated from a potential conductivity (ref to 0 dbar) multiplicative adjustment term r.                                                                                                                                  PRES_ADJUSTED = PRES                                                                                                                                                                                                                                            TEMP_ADJUSTED = TEMP                                                                                                                                                                                                                                            PSAL_ADJ = CTM_ADJ_PSAL + dS, dS is calculated from a potential conductivity (ref to 0 dbar) multiplicative adjustment term r.                                                                                                                                  PRES_ADJUSTED = PRES                                                                                                                                                                                                                                            TEMP_ADJUSTED = TEMP                                                                                                                                                                                                                                            PSAL_ADJ = CTM_ADJ_PSAL + dS, dS is calculated from a potential conductivity (ref to 0 dbar) multiplicative adjustment term r.                                                                                                                                  PRES_ADJUSTED = PRES                                                                                                                                                                                                                                            TEMP_ADJUSTED = TEMP                                                                                                                                                                                                                                            PSAL_ADJ = CTM_ADJ_PSAL + dS, dS is calculated from a potential conductivity (ref to 0 dbar) multiplicative adjustment term r.                                                                                                                                  PRES_ADJUSTED = PRES                                                                                                                                                                                                                                            TEMP_ADJUSTED = TEMP                                                                                                                                                                                                                                            PSAL_ADJ = CTM_ADJ_PSAL + dS, dS is calculated from a potential conductivity (ref to 0 dbar) multiplicative adjustment term r.                                                                                                                                  PRES_ADJUSTED = PRES                                                                                                                                                                                                                                            TEMP_ADJUSTED = TEMP                                                                                                                                                                                                                                            PSAL_ADJ = CTM_ADJ_PSAL + dS, dS is calculated from a potential conductivity (ref to 0 dbar) multiplicative adjustment term r.                                                                                                                                  None                                                                                                                                                                                                                                                            None                                                                                                                                                                                                                                                            CTM: alpha=0.141C, tau=6.89s, rise rate = 10 cm/s with error equal to the adjustment;OW: r =1.0002(+/-0.0005), vertically averaged dS =0.007(+/-0.018),800 < P < --,  Map Scales:[x=4,2; y=2,1].                                                                None                                                                                                                                                                                                                                                            None                                                                                                                                                                                                                                                            CTM: alpha=0.141C, tau=6.89s, rise rate = 10 cm/s with error equal to the adjustment;OW: r =1.0002(+/-0.0004), vertically averaged dS =0.009(+/-0.017),800 < P < --,  Map Scales:[x=4,2; y=2,1].                                                                None                                                                                                                                                                                                                                                            None                                                                                                                                                                                                                                                            CTM: alpha=0.141C, tau=6.89s, rise rate = 10 cm/s with error equal to the adjustment;OW: r =1.0003(+/-0.0004), vertically averaged dS =0.011(+/-0.015),800 < P < --,  Map Scales:[x=4,2; y=2,1].                                                                None                                                                                                                                                                                                                                                            None                                                                                                                                                                                                                                                            CTM: alpha=0.141C, tau=6.89s, rise rate = 10 cm/s with error equal to the adjustment;OW: r =1.0003(+/-0.0004), vertically averaged dS =0.014(+/-0.014),800 < P < --,  Map Scales:[x=4,2; y=2,1].                                                                None                                                                                                                                                                                                                                                            None                                                                                                                                                                                                                                                            CTM: alpha=0.141C, tau=6.89s, rise rate = 10 cm/s with error equal to the adjustment;OW: r =1.0004(+/-0.0003), vertically averaged dS =0.016(+/-0.013),800 < P < --,  Map Scales:[x=4,2; y=2,1].                                                                None                                                                                                                                                                                                                                                            None                                                                                                                                                                                                                                                            CTM: alpha=0.141C, tau=6.89s, rise rate = 10 cm/s with error equal to the adjustment;OW: r =1.0005(+/-0.0003), vertically averaged dS =0.018(+/-0.012),800 < P < --,  Map Scales:[x=4,2; y=2,1].                                                                None                                                                                                                                                                                                                                                            None                                                                                                                                                                                                                                                            CTM: alpha=0.141C, tau=6.89s, rise rate = 10 cm/s with error equal to the adjustment;OW: r =1.0005(+/-0.0003), vertically averaged dS =0.02(+/-0.011),800 < P < --,  Map Scales:[x=4,2; y=2,1].                                                                 None                                                                                                                                                                                                                                                            None                                                                                                                                                                                                                                                            CTM: alpha=0.141C, tau=6.89s, rise rate = 10 cm/s with error equal to the adjustment;OW: r =1.0006(+/-0.0003), vertically averaged dS =0.022(+/-0.01),800 < P < --,  Map Scales:[x=4,2; y=2,1].                                                                 None                                                                                                                                                                                                                                                            None                                                                                                                                                                                                                                                            CTM: alpha=0.141C, tau=6.89s, rise rate = 10 cm/s with error equal to the adjustment;OW: r =1.0006(+/-0.0002), vertically averaged dS =0.024(+/-0.01),800 < P < --,  Map Scales:[x=4,2; y=2,1].                                                                 None                                                                                                                                                                                                                                                            None                                                                                                                                                                                                                                                            CTM: alpha=0.141C, tau=6.89s, rise rate = 10 cm/s with error equal to the adjustment;OW: r =1.0007(+/-0.0002), vertically averaged dS =0.027(+/-0.009),800 < P < --,  Map Scales:[x=4,2; y=2,1].                                                                None                                                                                                                                                                                                                                                            None                                                                                                                                                                                                                                                            CTM: alpha=0.141C, tau=6.89s, rise rate = 10 cm/s with error equal to the adjustment;OW: r =1.0007(+/-0.0002), vertically averaged dS =0.029(+/-0.009),800 < P < --,  Map Scales:[x=4,2; y=2,1].                                                                None                                                                                                                                                                                                                                                            None                                                                                                                                                                                                                                                            CTM: alpha=0.141C, tau=6.89s, rise rate = 10 cm/s with error equal to the adjustment;OW: r =1.0008(+/-0.0002), vertically averaged dS =0.031(+/-0.008),800 < P < --,  Map Scales:[x=4,2; y=2,1].                                                                None                                                                                                                                                                                                                                                            None                                                                                                                                                                                                                                                            CTM: alpha=0.141C, tau=6.89s, rise rate = 10 cm/s with error equal to the adjustment;OW: r =1.0008(+/-0.0002), vertically averaged dS =0.033(+/-0.008),800 < P < --,  Map Scales:[x=4,2; y=2,1].                                                                None                                                                                                                                                                                                                                                            None                                                                                                                                                                                                                                                            NoneOW: r =1.0009(+/-0.0002), vertically averaged dS =NaN(+/-NaN),800 < P < --,  Map Scales:[x=4,2; y=2,1].                                                                                                                                                     None                                                                                                                                                                                                                                                            None                                                                                                                                                                                                                                                            NoneOW: r =1.0009(+/-0.0002), vertically averaged dS =NaN(+/-NaN),800 < P < --,  Map Scales:[x=4,2; y=2,1].                                                                                                                                                     None                                                                                                                                                                                                                                                            None                                                                                                                                                                                                                                                            CTM: alpha=0.141C, tau=6.89s, rise rate = 10 cm/s with error equal to the adjustment;OW: r =1.001(+/-0.0002), vertically averaged dS =0.04(+/-0.009),800 < P < --,  Map Scales:[x=4,2; y=2,1].                                                                  None                                                                                                                                                                                                                                                            None                                                                                                                                                                                                                                                            CTM: alpha=0.141C, tau=6.89s, rise rate = 10 cm/s with error equal to the adjustment;OW: r =1.0011(+/-0.0002), vertically averaged dS =0.042(+/-0.01),800 < P < --,  Map Scales:[x=4,2; y=2,1].                                                                 None                                                                                                                                                                                                                                                            None                                                                                                                                                                                                                                                            CTM: alpha=0.141C, tau=6.89s, rise rate = 10 cm/s with error equal to the adjustment;OW: r =1.0011(+/-0.0003), vertically averaged dS =0.044(+/-0.011),800 < P < --,  Map Scales:[x=4,2; y=2,1].                                                                None                                                                                                                                                                                                                                                            None                                                                                                                                                                                                                                                            CTM: alpha=0.141C, tau=6.89s, rise rate = 10 cm/s with error equal to the adjustment;OW: r =1.0012(+/-0.0003), vertically averaged dS =0.046(+/-0.011),800 < P < --,  Map Scales:[x=4,2; y=2,1].                                                                None                                                                                                                                                                                                                                                            None                                                                                                                                                                                                                                                            CTM: alpha=0.141C, tau=6.89s, rise rate = 10 cm/s with error equal to the adjustment;OW: r =1.0012(+/-0.0003), vertically averaged dS =0.048(+/-0.012),800 < P < --,  Map Scales:[x=4,2; y=2,1].                                                                None                                                                                                                                                                                                                                                            None                                                                                                                                                                                                                                                            CTM: alpha=0.141C, tau=6.89s, rise rate = 10 cm/s with error equal to the adjustment;OW: r =1.0013(+/-0.0003), vertically averaged dS =0.051(+/-0.013),800 < P < --,  Map Scales:[x=4,2; y=2,1].                                                                None                                                                                                                                                                                                                                                            None                                                                                                                                                                                                                                                            CTM: alpha=0.141C, tau=6.89s, rise rate = 10 cm/s with error equal to the adjustment;OW: r =1.0013(+/-0.0004), vertically averaged dS =0.053(+/-0.014),800 < P < --,  Map Scales:[x=4,2; y=2,1].                                                                None                                                                                                                                                                                                                                                            None                                                                                                                                                                                                                                                            CTM: alpha=0.141C, tau=6.89s, rise rate = 10 cm/s with error equal to the adjustment;OW: r =1.0014(+/-0.0004), vertically averaged dS =0.055(+/-0.015),800 < P < --,  Map Scales:[x=4,2; y=2,1].                                                                SOLO-W floats auto-correct mild pressure drift by zeroing the pressure sensor while on the surface.  Additional correction was unnecessary in DMQC;      PRES_ADJ_ERR: SBE sensor accuracy + resolution error                                                   No significant temperature drift detected;  TEMP_ADJ_ERR: SBE sensor accuracy + resolution error                                                                                                                                                                PSAL_ADJ corrects Conductivity Thermal Mass (CTM), Johnson et al., 2007, JAOT.; Significant salinity drift, OW fit adopted: fit for cycles 0 to 22.  The quoted error is max[0.01, 1xOW uncertainty] in PSS-78.                                                 SOLO-W floats auto-correct mild pressure drift by zeroing the pressure sensor while on the surface.  Additional correction was unnecessary in DMQC;      PRES_ADJ_ERR: SBE sensor accuracy + resolution error                                                   No significant temperature drift detected;  TEMP_ADJ_ERR: SBE sensor accuracy + resolution error                                                                                                                                                                PSAL_ADJ corrects Conductivity Thermal Mass (CTM), Johnson et al., 2007, JAOT.; Significant salinity drift, OW fit adopted: fit for cycles 0 to 22.  The quoted error is max[0.01, 1xOW uncertainty] in PSS-78.                                                 SOLO-W floats auto-correct mild pressure drift by zeroing the pressure sensor while on the surface.  Additional correction was unnecessary in DMQC;      PRES_ADJ_ERR: SBE sensor accuracy + resolution error                                                   No significant temperature drift detected;  TEMP_ADJ_ERR: SBE sensor accuracy + resolution error                                                                                                                                                                PSAL_ADJ corrects Conductivity Thermal Mass (CTM), Johnson et al., 2007, JAOT.; Significant salinity drift, OW fit adopted: fit for cycles 0 to 22.  The quoted error is max[0.01, 1xOW uncertainty] in PSS-78.                                                 SOLO-W floats auto-correct mild pressure drift by zeroing the pressure sensor while on the surface.  Additional correction was unnecessary in DMQC;      PRES_ADJ_ERR: SBE sensor accuracy + resolution error                                                   No significant temperature drift detected;  TEMP_ADJ_ERR: SBE sensor accuracy + resolution error                                                                                                                                                                PSAL_ADJ corrects Conductivity Thermal Mass (CTM), Johnson et al., 2007, JAOT.; Significant salinity drift, OW fit adopted: fit for cycles 0 to 22.  The quoted error is max[0.01, 1xOW uncertainty] in PSS-78.                                                 SOLO-W floats auto-correct mild pressure drift by zeroing the pressure sensor while on the surface.  Additional correction was unnecessary in DMQC;      PRES_ADJ_ERR: SBE sensor accuracy + resolution error                                                   No significant temperature drift detected;  TEMP_ADJ_ERR: SBE sensor accuracy + resolution error                                                                                                                                                                PSAL_ADJ corrects Conductivity Thermal Mass (CTM), Johnson et al., 2007, JAOT.; Significant salinity drift, OW fit adopted: fit for cycles 0 to 22.  The quoted error is max[0.01, 1xOW uncertainty] in PSS-78.                                                 SOLO-W floats auto-correct mild pressure drift by zeroing the pressure sensor while on the surface.  Additional correction was unnecessary in DMQC;      PRES_ADJ_ERR: SBE sensor accuracy + resolution error                                                   No significant temperature drift detected;  TEMP_ADJ_ERR: SBE sensor accuracy + resolution error                                                                                                                                                                PSAL_ADJ corrects Conductivity Thermal Mass (CTM), Johnson et al., 2007, JAOT.; Significant salinity drift, OW fit adopted: fit for cycles 0 to 22.  The quoted error is max[0.01, 1xOW uncertainty] in PSS-78.                                                 SOLO-W floats auto-correct mild pressure drift by zeroing the pressure sensor while on the surface.  Additional correction was unnecessary in DMQC;      PRES_ADJ_ERR: SBE sensor accuracy + resolution error                                                   No significant temperature drift detected;  TEMP_ADJ_ERR: SBE sensor accuracy + resolution error                                                                                                                                                                PSAL_ADJ corrects Conductivity Thermal Mass (CTM), Johnson et al., 2007, JAOT.; Significant salinity drift, OW fit adopted: fit for cycles 0 to 22.  The quoted error is max[0.01, 1xOW uncertainty] in PSS-78.                                                 SOLO-W floats auto-correct mild pressure drift by zeroing the pressure sensor while on the surface.  Additional correction was unnecessary in DMQC;      PRES_ADJ_ERR: SBE sensor accuracy + resolution error                                                   No significant temperature drift detected;  TEMP_ADJ_ERR: SBE sensor accuracy + resolution error                                                                                                                                                                PSAL_ADJ corrects Conductivity Thermal Mass (CTM), Johnson et al., 2007, JAOT.; Significant salinity drift, OW fit adopted: fit for cycles 0 to 22.  The quoted error is max[0.01, 1xOW uncertainty] in PSS-78.                                                 SOLO-W floats auto-correct mild pressure drift by zeroing the pressure sensor while on the surface.  Additional correction was unnecessary in DMQC;      PRES_ADJ_ERR: SBE sensor accuracy + resolution error                                                   No significant temperature drift detected;  TEMP_ADJ_ERR: SBE sensor accuracy + resolution error                                                                                                                                                                PSAL_ADJ corrects Conductivity Thermal Mass (CTM), Johnson et al., 2007, JAOT.; Significant salinity drift, OW fit adopted: fit for cycles 0 to 22.  The quoted error is max[0.01, 1xOW uncertainty] in PSS-78.                                                 SOLO-W floats auto-correct mild pressure drift by zeroing the pressure sensor while on the surface.  Additional correction was unnecessary in DMQC;      PRES_ADJ_ERR: SBE sensor accuracy + resolution error                                                   No significant temperature drift detected;  TEMP_ADJ_ERR: SBE sensor accuracy + resolution error                                                                                                                                                                PSAL_ADJ corrects Conductivity Thermal Mass (CTM), Johnson et al., 2007, JAOT.; Significant salinity drift, OW fit adopted: fit for cycles 0 to 22.  The quoted error is max[0.01, 1xOW uncertainty] in PSS-78.                                                 SOLO-W floats auto-correct mild pressure drift by zeroing the pressure sensor while on the surface.  Additional correction was unnecessary in DMQC;      PRES_ADJ_ERR: SBE sensor accuracy + resolution error                                                   No significant temperature drift detected;  TEMP_ADJ_ERR: SBE sensor accuracy + resolution error                                                                                                                                                                PSAL_ADJ corrects Conductivity Thermal Mass (CTM), Johnson et al., 2007, JAOT.; Significant salinity drift, OW fit adopted: fit for cycles 0 to 22.  The quoted error is max[0.01, 1xOW uncertainty] in PSS-78.                                                 SOLO-W floats auto-correct mild pressure drift by zeroing the pressure sensor while on the surface.  Additional correction was unnecessary in DMQC;      PRES_ADJ_ERR: SBE sensor accuracy + resolution error                                                   No significant temperature drift detected;  TEMP_ADJ_ERR: SBE sensor accuracy + resolution error                                                                                                                                                                PSAL_ADJ corrects Conductivity Thermal Mass (CTM), Johnson et al., 2007, JAOT.; Significant salinity drift, OW fit adopted: fit for cycles 0 to 22.  The quoted error is max[0.01, 1xOW uncertainty] in PSS-78.                                                 SOLO-W floats auto-correct mild pressure drift by zeroing the pressure sensor while on the surface.  Additional correction was unnecessary in DMQC;      PRES_ADJ_ERR: SBE sensor accuracy + resolution error                                                   No significant temperature drift detected;  TEMP_ADJ_ERR: SBE sensor accuracy + resolution error                                                                                                                                                                PSAL_ADJ corrects Conductivity Thermal Mass (CTM), Johnson et al., 2007, JAOT.; Significant salinity drift, OW fit adopted: fit for cycles 0 to 22.  The quoted error is max[0.01, 1xOW uncertainty] in PSS-78.                                                 SOLO-W floats auto-correct mild pressure drift by zeroing the pressure sensor while on the surface.  Additional correction was unnecessary in DMQC;      PRES_ADJ_ERR: SBE sensor accuracy + resolution error                                                   No significant temperature drift detected;  TEMP_ADJ_ERR: SBE sensor accuracy + resolution error                                                                                                                                                                No thermal lag adjustment because of insufficient data. Significant salinity drift, OW fit adopted: fit for cycles 0 to 22.  The quoted error is max[0.01, 1xOW uncertainty] in PSS-78.                                                                         SOLO-W floats auto-correct mild pressure drift by zeroing the pressure sensor while on the surface.  Additional correction was unnecessary in DMQC;      PRES_ADJ_ERR: SBE sensor accuracy + resolution error                                                   No significant temperature drift detected;  TEMP_ADJ_ERR: SBE sensor accuracy + resolution error                                                                                                                                                                No thermal lag adjustment because of insufficient data. Significant salinity drift, OW fit adopted: fit for cycles 0 to 22.  The quoted error is max[0.01, 1xOW uncertainty] in PSS-78.                                                                         SOLO-W floats auto-correct mild pressure drift by zeroing the pressure sensor while on the surface.  Additional correction was unnecessary in DMQC;      PRES_ADJ_ERR: SBE sensor accuracy + resolution error                                                   No significant temperature drift detected;  TEMP_ADJ_ERR: SBE sensor accuracy + resolution error                                                                                                                                                                PSAL_ADJ corrects Conductivity Thermal Mass (CTM), Johnson et al., 2007, JAOT.; Significant salinity drift, OW fit adopted: fit for cycles 0 to 22.  The quoted error is max[0.01, 1xOW uncertainty] in PSS-78.                                                 SOLO-W floats auto-correct mild pressure drift by zeroing the pressure sensor while on the surface.  Additional correction was unnecessary in DMQC;      PRES_ADJ_ERR: SBE sensor accuracy + resolution error                                                   No significant temperature drift detected;  TEMP_ADJ_ERR: SBE sensor accuracy + resolution error                                                                                                                                                                PSAL_ADJ corrects Conductivity Thermal Mass (CTM), Johnson et al., 2007, JAOT.; Significant salinity drift, OW fit adopted: fit for cycles 0 to 22.  The quoted error is max[0.01, 1xOW uncertainty] in PSS-78.                                                 SOLO-W floats auto-correct mild pressure drift by zeroing the pressure sensor while on the surface.  Additional correction was unnecessary in DMQC;      PRES_ADJ_ERR: SBE sensor accuracy + resolution error                                                   No significant temperature drift detected;  TEMP_ADJ_ERR: SBE sensor accuracy + resolution error                                                                                                                                                                PSAL_ADJ corrects Conductivity Thermal Mass (CTM), Johnson et al., 2007, JAOT.; Significant salinity drift, OW fit adopted: fit for cycles 0 to 22.  The quoted error is max[0.01, 1xOW uncertainty] in PSS-78.                                                 SOLO-W floats auto-correct mild pressure drift by zeroing the pressure sensor while on the surface.  Additional correction was unnecessary in DMQC;      PRES_ADJ_ERR: SBE sensor accuracy + resolution error                                                   No significant temperature drift detected;  TEMP_ADJ_ERR: SBE sensor accuracy + resolution error                                                                                                                                                                PSAL_ADJ corrects Conductivity Thermal Mass (CTM), Johnson et al., 2007, JAOT.; Significant salinity drift, OW fit adopted: fit for cycles 0 to 22.  The quoted error is max[0.01, 1xOW uncertainty] in PSS-78.                                                 SOLO-W floats auto-correct mild pressure drift by zeroing the pressure sensor while on the surface.  Additional correction was unnecessary in DMQC;      PRES_ADJ_ERR: SBE sensor accuracy + resolution error                                                   No significant temperature drift detected;  TEMP_ADJ_ERR: SBE sensor accuracy + resolution error                                                                                                                                                                PSAL_ADJ corrects Conductivity Thermal Mass (CTM), Johnson et al., 2007, JAOT.; Significant salinity drift, OW fit adopted: fit for cycles 0 to 22.  The quoted error is max[0.01, 1xOW uncertainty] in PSS-78.                                                 SOLO-W floats auto-correct mild pressure drift by zeroing the pressure sensor while on the surface.  Additional correction was unnecessary in DMQC;      PRES_ADJ_ERR: SBE sensor accuracy + resolution error                                                   No significant temperature drift detected;  TEMP_ADJ_ERR: SBE sensor accuracy + resolution error                                                                                                                                                                PSAL_ADJ corrects Conductivity Thermal Mass (CTM), Johnson et al., 2007, JAOT.; Significant salinity drift, OW fit adopted: fit for cycles 0 to 22.  The quoted error is max[0.01, 1xOW uncertainty] in PSS-78.                                                 SOLO-W floats auto-correct mild pressure drift by zeroing the pressure sensor while on the surface.  Additional correction was unnecessary in DMQC;      PRES_ADJ_ERR: SBE sensor accuracy + resolution error                                                   No significant temperature drift detected;  TEMP_ADJ_ERR: SBE sensor accuracy + resolution error                                                                                                                                                                PSAL_ADJ corrects Conductivity Thermal Mass (CTM), Johnson et al., 2007, JAOT.; Significant salinity drift, OW fit adopted: fit for cycles 0 to 22.  The quoted error is max[0.01, 1xOW uncertainty] in PSS-78.                                                 SOLO-W floats auto-correct mild pressure drift by zeroing the pressure sensor while on the surface.  Additional correction was unnecessary in DMQC;      PRES_ADJ_ERR: SBE sensor accuracy + resolution error                                                   No significant temperature drift detected;  TEMP_ADJ_ERR: SBE sensor accuracy + resolution error                                                                                                                                                                PSAL_ADJ corrects Conductivity Thermal Mass (CTM), Johnson et al., 2007, JAOT.; Significant salinity drift, OW fit adopted: fit for cycles 0 to 22.  The quoted error is max[0.01, 1xOW uncertainty] in PSS-78.                                                 202206010000002022060100000020220601000000202206010000002022060100000020220601000000202206010000002022060100000020220601000000202206010000002022060100000020220601000000202206010000002022060100000020220601000000202206010000002022060100000020220601000000202206010000002022060100000020220601000000202206010000002022060100000020220601000000202206010000002022060100000020220601000000202206010000002022060100000020220601000000202206010000002022060100000020220601000000202206010000002022060100000020220601000000202206010000002022060100000020220601000000202206010000002022060100000020220601000000202206010000002022060100000020220601000000202206010000002022060100000020220601000000202206010000002022060100000020220601000000202206010000002022060100000020220601000000202206010000002022060100000020220601000000202206010000002022060100000020220601000000202206010000002022060100000020220601000000202206010000002022060100000020220601000000202206010000002022060100000020220601000000  