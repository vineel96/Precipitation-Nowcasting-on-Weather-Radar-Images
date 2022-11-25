# Precipitation Nowcasting on Weather Radar Images
**Precipiation Nowcasting:** To make time specific and  position specific weather forecasting over a short period of X-hour(s) lead time for every Y-minutes interval.<br />
**Task:** Given input weather radar images from t-n,...,t-1,t timesteps, model will predict very next lead time weather radar images for timesteps t+1, t+2,...t+n
We deploy optical flow extrapolation with the help of U-Net network which outputs future optical flow information. We recursivly sample grid using previous frame
and extrapolated optical flow in order to construct frame at t+1, t+2,...,t+n timesteps.
![](https://github.com/vineel96/Precipitation-Nowcasting-on-Weather-Radar-Images/blob/main/nowcast-1.png)

# Datsets
## WeatherNews Inc Japan Weather Radar Images
We use weather radar image dataset provided by WeatherNews Inc Japan.  

# Implementation Details
Optical flow based extrapolation via U-Net:
- Trained on RAW radar data
- Training Data: 20190908 (upto 20190908_2030)  [ crop: (W,H): (401,721) ]
                          20190309 (upto 20190309_2030)  [ crop: (W,H): (961,1081) ]
                          20190703 ( Full day)                      [ crop: (W,H): (961,1081) ]
- Epochs: ~200
- Input: Optical flow information from 3 frames(i.e Computed optical flow between successive frames)
- Optical Flow Estimation: Dual TVL1
- Input Size:  512 * 512 * 4 
- Output: Extrapolated Optical Flow
- Output Size:  512 * 512 * 2
- Loss Function:  Lorentzian Loss + Divergence Loss + Vorticity Loss
- Post processing: In future frame construction from DL-OF result, we change the wrapping orientation in opposite direction 

# Model Training and Evaluation
- To train optical flow based U-net : python optical_flow_local.py
- To train unet + resnet variants : 
    - python unet+resnet_charbonnier.py
    - python unet+resnet_logcosh.py

# Acknowledgment
This research/project is supported in part by WeatherNews Inc Japan which provided us with their private dataset consisting of weather radat images
