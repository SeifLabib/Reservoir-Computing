disp('Clearing workspace.');
clear;

linewidth = 3;
fontsize = 14;
fontweight = 'bold';

N = 1000;
%p = 0.1;                % 10% sparsity
%p = 0.05;               % 5% sparsity
g = 1.5;				% g greater than 1 leads to chaotic networks.
alpha = 1.0;
nsecs = 3000; %5000; %3000; %2000; %1440;
dt = 0.1;
learn_every = 2;

simtime = 0:dt:nsecs-dt;
simtime_len = length(simtime);
simtime2 = 1*nsecs:dt:2*nsecs-dt;

amp = 1.3;
freq = 1/60;
ft = (amp/1.0)*sin(1.0*pi*freq*simtime) + ...
     (amp/2.0)*sin(2.0*pi*freq*simtime) + ...
     (amp/6.0)*sin(3.0*pi*freq*simtime) + ...
     (amp/3.0)*sin(4.0*pi*freq*simtime);
ft = ft/1.5;

ft2 = (amp/1.0)*sin(1.0*pi*freq*simtime2) + ...
      (amp/2.0)*sin(2.0*pi*freq*simtime2) + ...
      (amp/6.0)*sin(3.0*pi*freq*simtime2) + ...
      (amp/3.0)*sin(4.0*pi*freq*simtime2);
ft2 = ft2/1.5;

error = zeros(34, 1);
percentage = 0.01:0.03:1;
train_error = zeros(34, 1);


for j=0:33
p = j*0.03 + 0.01;

scale = 1.0/sqrt(p*N);
M = sprandn(N,N,p)*g*scale;
M = full(M);

nRec2Out = N;
wo = zeros(nRec2Out,1);
dw = zeros(nRec2Out,1);
wf = 2.0*(rand(N,1)-0.5);

wo_len = zeros(1,simtime_len);    
zt = zeros(1,simtime_len);
zpt = zeros(1,simtime_len);
x0 = 0.5*randn(N,1);
z0 = 0.5*randn(1,1);

x = x0; 
r = tanh(x);
z = z0; 

ti = 0;
P = (1.0/alpha)*eye(nRec2Out);
for t = simtime
    ti = ti+1;
    
    % sim, so x(t) and r(t) are created.
    x = (1.0-dt)*x + M*(r*dt) + wf*(z*dt);
    r = tanh(x);
    z = wo'*r;
    
    if mod(ti, learn_every) == 0
	% update inverse correlation matrix
	k = P*r;
	rPr = r'*k;
	c = 1.0/(1.0 + rPr);
	P = P - k*(k'*c);
    
	% update the error for the linear readout
	e = z-ft(ti);
	
	% update the output weights
	dw = -e*k*c;	
	wo = wo + dw;
    end
    
    % Store the output of the system.
    zt(ti) = z;
    wo_len(ti) = sqrt(wo'*wo);	
end
error_avg = sum(abs(zt-ft))/simtime_len;
disp(['Training MAE: ' num2str(error_avg,3)]);    
disp(['Now testing... please wait.']);
train_error(j+1) = abs(1- error_avg);

% Now test. 
ti = 0;
zeroIndices = randperm(N, 1);

for t = simtime				% don't want to subtract time in indices
    ti = ti+1;

    % sim, so x(t) and r(t) are created
    x = (1.0-dt)*x + M*(r*dt) + wf*(z*dt);
    x(zeroIndices) = 0;
    r = tanh(x);
    z = wo'*r;
    
    zpt(ti) = z;
end
error_avg = sum(abs(zpt-ft2))/simtime_len;
disp(['Testing MAE: ' num2str(error_avg,3)]);

disp(['j =' num2str(j)]);
error(j+1) = abs(1-error_avg);
end

% Compute the FFT
N = length(ft2); % Number of points in the signal
Fs = 1/(simtime(2)-simtime(1)); % Sampling frequency
Y = fft(ft2); % Compute the FFT
P2 = abs(Y/N); % Two-sided spectrum
P1 = P2(1:N/2+1); % Single-sided spectrum
P1(2:end-1) = 2*P1(2:end-1);

% Compute the frequency vector
f = Fs*(0:(N/2))/N;

% Find the dominant frequency
[~, idx] = max(P1);
dominantFrequency = f(idx);

% Find the maximum and minimum amplitudes
maxAmplitude = max(ft2);
minAmplitude = min(ft2);

disp(['desired output']);
disp(['min: ' num2str(minAmplitude, 3)]);
disp(['max:' num2str(maxAmplitude, 3)]);
disp(['dom frequ: ' num2str(dominantFrequency,5)]);

% Compute the FFT
N = length(zpt); % Number of points in the signal
Fs = 1/(simtime(2)-simtime(1)); % Sampling frequency
Y = fft(zpt); % Compute the FFT
P2 = abs(Y/N); % Two-sided spectrum
P1 = P2(1:N/2+1); % Single-sided spectrum
P1(2:end-1) = 2*P1(2:end-1);

% Compute the frequency vector
f = Fs*(0:(N/2))/N;

% Find the dominant frequency
[~, idx] = max(P1);
dominantFrequency = f(idx);

% Find the maximum and minimum amplitudes
maxAmplitude = max(zpt);
minAmplitude = min(zpt);

disp(['generated output']);
disp(['min: ' num2str(minAmplitude, 3)]);
disp(['max:' num2str(maxAmplitude, 3)]);
disp(['dom frequ: ' num2str(dominantFrequency,5)]);


figure;
subplot 211;
plot(percentage, train_error, 'linewidth', linewidth, 'color', 'green');
hold on;
plot(percentage, error, 'linewidth', linewidth, 'color', 'red');
title('Pattern similarity vs network connectivity sparsity', 'fontsize', fontsize-5, 'fontweight', fontweight);
xlabel('Connectivity sparsity', 'fontsize', fontsize-5, 'fontweight', fontweight);
hold on;
ylabel('Pattern similarity', 'fontsize', fontsize-5, 'fontweight', fontweight);
legend('training similarity', 'testing similarity');