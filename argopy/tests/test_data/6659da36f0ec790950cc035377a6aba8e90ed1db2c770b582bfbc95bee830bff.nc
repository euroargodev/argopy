CDF       
      row    7   data_mode_strlen      direction_strlen      platform_number_strlen        position_qc_strlen        pres_adjusted_qc_strlen       pres_qc_strlen        psal_adjusted_qc_strlen       psal_qc_strlen        temp_adjusted_qc_strlen       temp_qc_strlen        time_qc_strlen        vertical_sampling_scheme_strlen    �      $   cdm_altitude_proxy        pres   cdm_data_type         TrajectoryProfile      cdm_profile_variables        scycle_number, data_type, format_version, handbook_version, reference_date_time, date_creation, date_update, direction, data_center, dc_reference, data_state_indicator, data_mode, firmware_version, wmo_inst_type, time, time_qc, time_location, latitude, longitude, position_qc, positioning_system, profile_pres_qc, profile_temp_qc, profile_psal_qc, vertical_sampling_scheme    cdm_trajectory_variables      Fplatform_number, project_name, pi_name, platform_type, float_serial_no     Conventions       "Argo-3.1, CF-1.6, COARDS, ACDD-1.3     creator_email         support@argo.net   creator_name      Argo   creator_url       https://argo.ucsd.edu/     defaultGraphQuery         �longitude%2Clatitude%2Ctemp&time>=now-2d&time<=now&pres>=0&pres<=5&.draw=markers&.marker=5|5&.color=0x000000&.colorBar=|||||&.bgColor=0xffccccff   Easternmost_Easting       �0p��
>   featureType       TrajectoryProfile      geospatial_lat_max        ?�KƧ   geospatial_lat_min        ?�333334   geospatial_lat_units      degrees_north      geospatial_lon_max        �0p��
>   geospatial_lon_min        �0d�t�k   geospatial_lon_units      degrees_east   history      ~2025-03-04T09:37:14Z (local files)
2025-03-04T09:37:14Z https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.nc?config_mission_number,cycle_number,data_mode,direction,latitude,longitude,platform_number,position_qc,pres,pres_adjusted,pres_adjusted_error,pres_adjusted_qc,pres_qc,psal,psal_adjusted,psal_adjusted_error,psal_adjusted_qc,psal_qc,temp,temp_adjusted,temp_adjusted_error,temp_adjusted_qc,temp_qc,time,time_qc,vertical_sampling_scheme&longitude%3E=-20&longitude%3C=-16.0&latitude%3E=0&latitude%3C=1&pres%3E=0&pres%3C=100.0&time%3E=1072915200.0&time%3C=1075507200.0&latitude!=NaN&longitude!=NaN&distinct()&orderBy(%22time,pres%22)     id        
ArgoFloats     infoUrl       https://argo.ucsd.edu/     institution       Argo   keywords     �adjusted, argo, array, assembly, best, centre, centres, charge, coded, CONFIG_MISSION_NUMBER, contains, coriolis, creation, currents, cycle, CYCLE_NUMBER, data, DATA_CENTRE, DATA_MODE, DATA_STATE_INDICATOR, DATA_TYPE, date, DATE_CREATION, DATE_UPDATE, day, days, DC_REFERENCE, degree, delayed, denoting, density, determined, direction, Earth Science > Oceans > Ocean Pressure > Water Pressure, Earth Science > Oceans > Ocean Temperature > Water Temperature, Earth Science > Oceans > Salinity/Density > Salinity, equals, error, estimate, file, firmware, FIRMWARE_VERSION, flag, float, FLOAT_SERIAL_NO, format, FORMAT_VERSION, gdac, geostrophic, global, handbook, HANDBOOK_VERSION, have, identifier, in-situ, instrument, investigator, its, its-90, JULD, JULD_LOCATION, JULD_QC, julian, latitude, level, longitude, missions, mode, name, number, ocean, oceanography, oceans, passed, performed, PI_NAME, PLATFORM_NUMBER, PLATFORM_TYPE, position, POSITION_QC, positioning, POSITIONING_SYSTEM, practical, pres, PRES_ADJUSTED, PRES_ADJUSTED_ERROR, PRES_ADJUSTED_QC, PRES_QC, pressure, principal, process, processing, profile, PROFILE_PRES_QC, PROFILE_PSAL_QC, PROFILE_TEMP_QC, profiles, project, PROJECT_NAME, psal, PSAL_ADJUSTED, PSAL_ADJUSTED_ERROR, PSAL_ADJUSTED_QC, PSAL_QC, quality, rdac, real, real time, real-time, realtime, reference, REFERENCE_DATE_TIME, regional, relative, salinity, sampling, scale, scheme, sea, sea level, sea-level, sea_water_practical_salinity, sea_water_pressure, sea_water_temperature, seawater, serial, situ, station, statistics, system, TEMP, TEMP_ADJUSTED, TEMP_ADJUSTED_ERROR, TEMP_ADJUSTED_QC, TEMP_QC, temperature, through, time, type, unique, update, values, version, vertical, VERTICAL_SAMPLING_SCHEME, water, WMO_INST_TYPE   keywords_vocabulary       GCMD Science Keywords      license       falsestandard]     Northernmost_Northing         ?�KƧ   
references        (http://www.argodatamgt.org/Documentation   source        
Argo float     	sourceUrl         (local files)      Southernmost_Northing         ?�333334   standard_name_vocabulary      CF Standard Name Table v29     summary      �Argo float vertical profiles from Coriolis Global Data Assembly Centres
(GDAC). Argo is an international collaboration that collects high-quality
temperature and salinity profiles from the upper 2000m of the ice-free
global ocean and currents from intermediate depths. The data come from
battery-powered autonomous floats that spend most of their life drifting
at depth where they are stabilised by being neutrally buoyant at the
"parking depth" pressure by having a density equal to the ambient pressure
and a compressibility that is less than that of sea water. At present there
are several models of profiling float used in Argo. All work in a similar
fashion but differ somewhat in their design characteristics. At typically
10-day intervals, the floats pump fluid into an external bladder and rise
to the surface over about 6 hours while measuring temperature and salinity.
Satellites or GPS determine the position of the floats when they surface,
and the floats transmit their data to the satellites. The bladder then
deflates and the float returns to its original density and sinks to drift
until the cycle is repeated. Floats are designed to make about 150 such
cycles.
Data Management URL: http://www.argodatamgt.org/Documentation    time_coverage_end         2004-01-24T05:04:00Z   time_coverage_start       2004-01-04T05:10:00Z   title         Argo Float Measurements    user_manual_version       3.1    Westernmost_Easting       �0d�t�k         config_mission_number                   
_FillValue         ��   actual_range               colorBarMaximum       @Y         colorBarMinimum                  conventions       !1...N, 1 : first complete mission      ioos_category         
Statistics     	long_name         :Unique number denoting the missions performed by the float        �  =�   cycle_number                
_FillValue         ��   actual_range               cf_role       
profile_id     colorBarMaximum       @i         colorBarMinimum                  conventions       =0...N, 0 : launch cycle (if exists), 1 : first complete cycle      ioos_category         
Statistics     	long_name         Float cycle number        �  >�   	data_mode                      	_Encoding         
ISO-8859-1     conventions       >R : real time; D : delayed mode; A : real time with adjustment     ioos_category         Time   	long_name         Delayed mode or real time data        8  ?|   	direction                      	_Encoding         
ISO-8859-1     colorBarMaximum       @v�        colorBarMinimum                  conventions       -A: ascending profiles, D: descending profiles      ioos_category         Currents   	long_name         !Direction of the station profiles         8  ?�   latitude                _CoordinateAxisType       Lat    
_FillValue        @�i�       actual_range      ?�333334?�KƧ   axis      Y      colorBarMaximum       @V�        colorBarMinimum       �V�        ioos_category         Location   	long_name         &Latitude of the station, best estimate     standard_name         latitude   units         degrees_north      	valid_max         @V�        	valid_min         �V�          �  ?�   	longitude                   _CoordinateAxisType       Lon    
_FillValue        @�i�       actual_range      �0d�t�k�0p��
>   axis      X      colorBarMaximum       @f�        colorBarMinimum       �f�        ioos_category         Location   	long_name         'Longitude of the station, best estimate    standard_name         	longitude      units         degrees_east   	valid_max         @f�        	valid_min         �f�          �  A�   platform_number                    	_Encoding         
ISO-8859-1     cf_role       trajectory_id      conventions       WMO float identifier : A9IIIII     ioos_category         
Identifier     	long_name         Float unique identifier      �  C\   position_qc                    	_Encoding         
ISO-8859-1     colorBarMaximum       @b�        colorBarMinimum                  conventions       Argo reference table 2     ioos_category         Quality    	long_name         ,Quality on position (latitude and longitude)      8  D�   pres                _CoordinateAxisType       Height     
_FillValue        G�O�   actual_range      A   B�     axis      Z      C_format      %7.1f      colorBarMaximum       @��        colorBarMinimum                  FORTRAN_format        F7.1   ioos_category         	Sea Level      	long_name         )Sea water pressure, equals 0 at sea-level      sdn_parameter_urn         SDN:P01::PRESPR01      standard_name         sea_water_pressure     units         decibar    	valid_max         F;�    	valid_min                   �  E   pres_adjusted                   
_FillValue        G�O�   actual_range      A   B�     axis      Z      C_format      %7.1f      colorBarMaximum       @��        colorBarMinimum                  FORTRAN_format        F7.1   ioos_category         	Sea Level      	long_name         )Sea water pressure, equals 0 at sea-level      standard_name         sea_water_pressure     units         decibar    	valid_max         F;�    	valid_min                   �  E�   pres_adjusted_error              	   
_FillValue        G�O�   actual_range      @�  @�     C_format      %7.1f      colorBarMaximum       @I         colorBarMinimum                  FORTRAN_format        F7.1   ioos_category         
Statistics     	long_name         VContains the error on the adjusted values as determined by the delayed mode QC process     units         decibar       �  F�   pres_adjusted_qc                   	_Encoding         
ISO-8859-1     colorBarMaximum       @b�        colorBarMinimum                  conventions       Argo reference table 2     ioos_category         Quality    	long_name         quality flag      8  G�   pres_qc                    	_Encoding         
ISO-8859-1     colorBarMaximum       @b�        colorBarMinimum                  conventions       Argo reference table 2     ioos_category         Quality    	long_name         quality flag      8  G�   psal                
_FillValue        G�O�   actual_range      B�Bm�   C_format      %9.3f      colorBarMaximum       @B�        colorBarMinimum       @@         FORTRAN_format        F9.3   ioos_category         Salinity   	long_name         Practical salinity     sdn_parameter_urn         SDN:P01::PSALST01      standard_name         sea_water_practical_salinity   units         PSU    	valid_max         B$     	valid_min         @         �  H   psal_adjusted                   
_FillValue        G�O�   actual_range      B��B�   C_format      %9.3f      colorBarMaximum       @B�        colorBarMinimum       @@         FORTRAN_format        F9.3   ioos_category         Salinity   	long_name         Practical salinity     standard_name         sea_water_practical_salinity   units         PSU    	valid_max         B$     	valid_min         @         �  H�   psal_adjusted_error              	   
_FillValue        G�O�   actual_range      <#�
<#�
   C_format      %9.3f      colorBarMaximum       ?�         colorBarMinimum                  FORTRAN_format        F9.3   ioos_category         
Statistics     	long_name         VContains the error on the adjusted values as determined by the delayed mode QC process     units         psu       �  I�   psal_adjusted_qc                   	_Encoding         
ISO-8859-1     colorBarMaximum       @b�        colorBarMinimum                  conventions       Argo reference table 2     ioos_category         Quality    	long_name         quality flag      8  J�   psal_qc                    	_Encoding         
ISO-8859-1     colorBarMaximum       @b�        colorBarMinimum                  conventions       Argo reference table 2     ioos_category         Quality    	long_name         quality flag      8  J�   temp                
_FillValue        G�O�   actual_range      A{O�A���   C_format      %9.3f      colorBarMaximum       @@         colorBarMinimum                  FORTRAN_format        F9.3   ioos_category         Temperature    	long_name         $Sea temperature in-situ ITS-90 scale   sdn_parameter_urn         SDN:P01::TEMPST01      standard_name         sea_water_temperature      units         degree_Celsius     	valid_max         B      	valid_min         �         �  K    temp_adjusted                   
_FillValue        G�O�   actual_range      A{O�A���   C_format      %9.3f      colorBarMaximum       @@         colorBarMinimum                  FORTRAN_format        F9.3   ioos_category         Temperature    	long_name         $Sea temperature in-situ ITS-90 scale   standard_name         sea_water_temperature      units         degree_Celsius     	valid_max         B      	valid_min         �         �  K�   temp_adjusted_error              	   
_FillValue        G�O�   actual_range      <#�
<#�
   C_format      %9.3f      colorBarMaximum       ?�         colorBarMinimum                  FORTRAN_format        F9.3   ioos_category         
Statistics     	long_name         VContains the error on the adjusted values as determined by the delayed mode QC process     units         degree_Celsius        �  L�   temp_adjusted_qc          	         	_Encoding         
ISO-8859-1     colorBarMaximum       @b�        colorBarMinimum                  conventions       Argo reference table 2     ioos_category         Quality    	long_name         quality flag      8  M�   temp_qc           
         	_Encoding         
ISO-8859-1     colorBarMaximum       @b�        colorBarMinimum                  conventions       Argo reference table 2     ioos_category         Quality    	long_name         quality flag      8  M�   time                _CoordinateAxisType       Time   actual_range      A���   A�0      axis      T      ioos_category         Time   	long_name         ?Julian day (UTC) of the station relative to REFERENCE_DATE_TIME    standard_name         time   time_origin       01-JAN-1970 00:00:00   units         "seconds since 1970-01-01T00:00:00Z       �  N$   time_qc                    	_Encoding         
ISO-8859-1     colorBarMaximum       @b�        colorBarMinimum                  conventions       Argo reference table 2     ioos_category         Quality    	long_name         Quality on date and time      8  O�   vertical_sampling_scheme                   	_Encoding         
ISO-8859-1     conventions       Argo reference table 16    ioos_category         Unknown    	long_name         Vertical sampling scheme     �  P                                                                                                                                                                                                                                                                                                                                          DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA ?�KƧ?�KƧ?�KƧ?�KƧ?�KƧ?�KƧ?�KƧ?�KƧ?�KƧ?�KƧ?�KƧ?�KƧ?�KƧ?�KƧ?�KƧ?�KƧ?�KƧ?�KƧ?�KƧ?�;dZ�	?�;dZ�	?�;dZ�	?�;dZ�	?�;dZ�	?�;dZ�	?�;dZ�	?�;dZ�	?�;dZ�	?�;dZ�	?�;dZ�	?�;dZ�	?�;dZ�	?�;dZ�	?�;dZ�	?�;dZ�	?�;dZ�	?�;dZ�	?�333334?�333334?�333334?�333334?�333334?�333334?�333334?�333334?�333334?�333334?�333334?�333334?�333334?�333334?�333334?�333334?�333334?�333334�0&$�/��0&$�/��0&$�/��0&$�/��0&$�/��0&$�/��0&$�/��0&$�/��0&$�/��0&$�/��0&$�/��0&$�/��0&$�/��0&$�/��0&$�/��0&$�/��0&$�/��0&$�/��0&$�/��0p��
>�0p��
>�0p��
>�0p��
>�0p��
>�0p��
>�0p��
>�0p��
>�0p��
>�0p��
>�0p��
>�0p��
>�0p��
>�0p��
>�0p��
>�0p��
>�0p��
>�0p��
>�0d�t�k�0d�t�k�0d�t�k�0d�t�k�0d�t�k�0d�t�k�0d�t�k�0d�t�k�0d�t�k�0d�t�k�0d�t�k�0d�t�k�0d�t�k�0d�t�k�0d�t�k�0d�t�k�0d�t�k�0d�t�k1900207190020719002071900207190020719002071900207190020719002071900207190020719002071900207190020719002071900207190020719002071900207190020719002071900207190020719002071900207190020719002071900207190020719002071900207190020719002071900207190020719002071900207190020719002071900207190020719002071900207190020719002071900207190020719002071900207190020719002071900207190020719002071900207   1111111111111111111111111111111111111111111111111111111 A   AP  A�  A�  A�  B  B  B(  B<  BT  Bh  B|  B�  B�  B�  B�  B�  B�  B�  A@  A�  A�  A�  B   B  B,  B<  BP  Bd  Bx  B�  B�  B�  B�  B�  B�  B�  AP  A�  A�  A�  B  B  B,  B<  BP  Bh  B|  B�  B�  B�  B�  B�  B�  B�  A   AP  A�  A�  A�  B  B  B(  B<  BT  Bh  B|  B�  B�  B�  B�  B�  B�  B�  A@  A�  A�  A�  B   B  B,  B<  BP  Bd  Bx  B�  B�  B�  B�  B�  B�  B�  AP  A�  A�  A�  B  B  B,  B<  BP  Bh  B|  B�  B�  B�  B�  B�  B�  B�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  1111111111111111111111111111111111111111111111111111111 1111111111111111111111111111111111111111111111111111111 B��B��B��B�B�B�5B�B�B^5B��B�XB�?B�By�B�Bp�B{B	7B��B�BJB��BF�B!�B��B��B?}B�-B�FB��Bo�Bn�BYB6FB�B%B�)B��B��BȴBȴB��B��B�/BbNB��B �Bm�B�B��BB�B��Bq�BZG�O�B�BPB�BJ�B�B%sBP�B�.B��B�YB�;B�B��BTxB��BH�B=VB3B��B<1BɵBw BRyB��B�Bp�B�:B�RBʑB��B��B�BgBKkB6�B�B��B��B��B��B��BB
PB��B��BN4B�BHB��B/LB�BưB��B�0G�O�<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
4333333333333333333111111111111111111111111111111111111 4333333333333333333333333333333333333333333333333333333 A�=qA�9XA�$�AݬA�  A֩�A�VA͑hA�v�Aʧ�A�/A�|�A��wA�;dA��PA��A�jA�$�A{O�A���A�`BA�{A�JA�=qA�7LAƲ-A�ĜA�VA���A�A�A���A��\A�&�A�ƨA��HA��/A�?}AܑhA܏\AܓuA܍PA�\)A���A� �A��AμjA��`AǍPA���A�|�A��A��A���A�n�A�VG�O�A�9XA�$�AݬA�  A֩�A�VA͑hA�v�Aʧ�A�/A�|�A��wA�;dA��PA��A�jA�$�A{O�A���A�`BA�{A�JA�=qA�7LAƲ-A�ĜA�VA���A�A�A���A��\A�&�A�ƨA��HA��/A�?}AܑhA܏\AܓuA܍PA�\)A���A� �A��AμjA��`AǍPA���A�|�A��A��A���A�n�A�VG�O�<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
<#�
4111111111111111111111111111111111111111111111111111111 4111111111111111111111111111111111111111111111111111111 A���   A���   A���   A���   A���   A���   A���   A���   A���   A���   A���   A���   A���   A���   A���   A���   A���   A���   A���   A�2�   A�2�   A�2�   A�2�   A�2�   A�2�   A�2�   A�2�   A�2�   A�2�   A�2�   A�2�   A�2�   A�2�   A�2�   A�2�   A�2�   A�2�   A�0   A�0   A�0   A�0   A�0   A�0   A�0   A�0   A�0   A�0   A�0   A�0   A�0   A�0   A�0   A�0   A�0   A�0   1111111111111111111111111111111111111111111111111111111 Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]Primary sampling: averaged [10 sec sampling, 25 dbar average from 2000 dbar to 150 dbar; 10 sec sampling, 5 dbar average from 150 dbar to 7.5 dbar]   