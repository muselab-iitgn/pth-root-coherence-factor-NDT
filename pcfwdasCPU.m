clc; close all; clear all;
rng(1);
rawdata=importdata('b_mode_NDT_FMC_10V2025-10-22-p10.mat');
no_elements=rawdata.Trans.numelements;
signal_length = rawdata.Receive(1).endSample;
        data = squeeze(rawdata.RcvData{1}(:,:,1));

        for k = 1:no_elements
            shot = data(1:signal_length*no_elements,k);
            shot = (reshape(shot,signal_length,[]));
            data_matrix(:,:,k) = (double(shot)); % fliplr
        end
        paramWaveform.signals = shiftdim(data_matrix,2);
        paramWaveform.samplingFreq = rawdata.Receive(1).decimSampleRate;
        paramWaveform.filtering = 0;
%% --- Load Data and Extract Parameters ---
% [paramWaveform,paramWedge,paramPA,paramMaterial,paramTFM] = loadSampleSignals(1);

signals = paramWaveform.signals;  % [64 x 1397 x 64]
[num_tx, signal_length, num_rx] = size(signals);
% fprintf('Signal dimensions: [%d (TX) x %d (time) x %d (RX)]\n', num_tx, signal_length, num_rx);
% num_tx=64;num_rx=64;
% signal_length=1397;
%% --- Define Basic Parameters ---
element_spacing = 0.5; % paramPA.pitch;     % mm
xStep = element_spacing * 1e-3;        % m

% Sampling frequency (Hz) and speed of sound (m/s)
fs = 19.23*1e6; % paramWaveform.samplingFreq * 1e6; 
c = 6.32*1e3; % paramMaterial.c1L * 1e3;           
% Acquisition start delay (if any), set tDelay = 0 if not used.
tDelay = 0;

%% --- Time and Depth Axes ---
% Create time axis (in seconds)
t = tDelay + (0:signal_length-1)/fs;
% Depth (pulse-echo): converting time to depth (m)
depth = (t * c) / 2;
depth_axis_mm = depth * 1e3;  % Depth axis in mm

%% --- Define the Image Grid ---
% Define the lateral (x) grid covering the array aperture.
x_range = [-20:.1:20]*1000;
% Define axial (z) grid. (For example, from 0 to 60 mm.)
z_range = [10:.1:80]*1000;

tem_pcf(length(x_range), length(z_range)) = 0;
temp(length(x_range), length(z_range)) = 0;

[X, Z] = meshgrid(x_range, z_range);
[nZ, nX] = size(X);  % image grid dimensions

%% --- Compute Receive Element Positions and Delays ---
% Assume a linear array centered at zero for receive elements.
element_pos = ((0:num_rx-1) - (num_rx-1)/2) * xStep;  % in meters

% For a steering angle theta; for broadside, theta = 0.
theta = 0;  
% Compute delay for each element (in seconds)
tau = element_pos * sind(theta) / c;  
sample_delays = tau * fs;  % fractional sample delays in samples

%% --- Beamforming ---
tic
% Co = 8;  %Element configuration %%%% 1>>64,2>>32, 4>>16, 8>>8, 16>>4, 32>>2
p = 1;
N_elements=32; 

ele=randperm(numel(1:1:32),numel(1:N_elements));
Sortarrays=sort(ele,'ascend');
l = Sortarrays;
loc = linspace(-9.53, 9.53, 32) .* 1e-3;
loc = loc(Sortarrays);


% l = 1:Co:64;
% loc = linspace(-19.53, 19.53, 64) .* 1e-3;
% loc = loc(1:Co:64);
element_Pos_Array_um_X = loc .* 1e6;
RF_Start_Time = 0;
speed_Of_Sound_umps = 6354 * 10^6;
% Processing loop
for j = 1:length(l)
    disp(['Processing: ', num2str(j)]);
    rf_Data = (signals(:, 512 * l(j) - 511:512 * l(j)));
    rf_Data=fliplr(rf_Data');
    rf_Data = rf_Data(:, Sortarrays);  % 1:Co:32 for uniform array
    Single_element_loc=[element_Pos_Array_um_X(j),0];
    % Beamforming
%     [I] = PDAS(rf_Data, element_Pos_Array_um_X, speed_Of_Sound_umps, RF_Start_Time, fs, BeamformX, BeamformZ, Single_element_loc, p);
    [I] = pthcoherenceNDT(rf_Data, element_Pos_Array_um_X, speed_Of_Sound_umps, RF_Start_Time,fs,x_range, z_range, Single_element_loc,p);
    temp = abs(hilbert(I));
    tem_pcf = tem_pcf + temp;
    % Visualization (Existing)
    FrameData = abs(hilbert(I));
    tempx = (FrameData ./ (max(FrameData(:))));
    figure(2);
    imagesc(x_range*1e-3,z_range*1e-3,(tempx)'); axis equal; axis tight; colormap('hot'); colorbar('Direction','normal'); %caxis([-40 0]);
    C=colorbar;
    C.Label.String='Amplitude(in dB)';
    set(C.Label,'Position',[5,-10.668874227528,0]);
    set(C.Label,'FontSize',16)
    C.Label.Rotation=270;
    %C.Label.Direction='reverse';
    xlabel('Lateral Location (mm)');
    ylabel('Range Location (mm)');
    title('TFM using DAS');
    set(gca,'fontsize', 16);
%cd('D:\Shaswata\NDT data\Sparseimages\HPO\8 elements');
%cd('D:\Beamforming\Literature_Review\NDT_Coding\New_abstract_updated\White_background_VPO\64_elements\P4');
end
I=((tem_pcf'));
toc
%%
%load('64VPODAS1.mat')
%I=I;
%temp=abs(hilbert(I));
tempx=(I./(max(I(:))));
%tempfilt=bandpass(tempx,[1.85 2.15],fs);
% figure(3);
% imagesc(BeamformX*1e-3,BeamformZ*1e-3,(tempx)); axis equal; axis tight; colormap('hot'); colorbar('Direction','normal'); caxis([0 1])
% C=colorbar;
% C.Label.String='Amplitude(in dB)';
% set(C.Label,'Position',[5,-10.668874227528,0]);
% set(C.Label,'FontSize',22)
% C.Label.Rotation=270;
% %C.Label.Direction='reverse';
% xlabel('Lateral Location (mm)');
% ylabel('Range Location (mm)');
% title(' TFM using DAS');
% set(gca,'fontsize', 20);
figure(4);
imagesc(x_range * 1e-3, z_range * 1e-3, (tempx));
axis equal; 
axis tight;
colorbar('Direction', 'normal'); 
caxis([0 1]);
% Apply custom colormap
% Normalize the data for colormap scaling
%tempx = (I - min(I(:))) / (max(I(:)) - min(I(:)));
% Define a custom colormap with a faded transition from white to blue to green to yellow to red
custom_colormap = [
    1 1 1;    % White
    0 0 1;    % Blue
    0 1 0;    % Green
    1 1 0;    % Yellow
    1 0 0     % Red
];
% Create a colormap with interpolated colors
n = 64;  % Number of colormap entries
colormap_entries = size(custom_colormap, 1);
interp_colormap = interp1(1:colormap_entries, custom_colormap, linspace(1, colormap_entries, n));
% Apply the custom colormap
colormap(interp_colormap);
% Colorbar settings
C = colorbar;
C.Label.String = 'Amplitude (in dB)';
set(C.Label, 'Position', [5, -10.668874227528, 0]);
set(C.Label, 'FontSize', 22);
C.Label.Rotation = 270;

% Axes labels and title
xlabel('Lateral Location (mm)');
ylabel('Range Location (mm)');
% title('TFM using DAS with Custom Colormap');
set(gca, 'fontsize', 16);

% %% Save the figure and data for figure(4)
% temp_Image = ['pCFwDAS5_8P10','.fig'];
% saveas(gca, temp_Image);
% save('pCFwDAS5_8P10', 'I')

