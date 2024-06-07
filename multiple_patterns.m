% multiple_patterns.m
%
% This function generates the sum of 4 sine waves in figure 2D using the architecture of figure 1A with the RLS
% learning rule.
% The model was modified in order to train the same readout layer neuron on
% different patterns
%
% written by David Sussillo & Seif Labib

%

%% Training %%
disp('Clearing workspace.');
clear;

linewidth = 3;
fontsize = 14;
fontweight = 'bold';

N = 1000;
p = 0.1;
g = 1.5;				% g greater than 1 leads to chaotic networks.
alpha = 1.0;
nsecs = 5000; %3000; %2000; %1440;
dt = 0.1;
learn_every = 2;

chosen = 0;

scale = 1.0/sqrt(p*N);
M = sprandn(N,N,p)*g*scale;
M = full(M);

N_patterns = 2;
nRec2Out = N;
wo = zeros(nRec2Out,N_patterns);
dw = zeros(nRec2Out,1);
wf = 2.0*(rand(N,1)-0.5);


simtime = 0:dt:nsecs-dt;
simtime_len = length(simtime);
simtime2 = 1*nsecs:dt:2*nsecs-dt;

amp = 1.3;
freq = 1/60;
ft1 = (amp/1.0)*sin(1.0*pi*freq*simtime) + ...
     (amp/2.0)*sin(2.0*pi*freq*simtime) + ...
     (amp/6.0)*sin(3.0*pi*freq*simtime) + ...
     (amp/3.0)*sin(4.0*pi*freq*simtime);
ft1 = ft1/1.5;

ft2 = (amp/1.0)*cos(1.0*pi*freq*simtime2) + ...
      (amp/2.0)*cos(2.0*pi*freq*simtime2) + ...
      (amp/6.0)*cos(3.0*pi*freq*simtime2) + ...
      (amp/3.0)*cos(4.0*pi*freq*simtime2);
ft2 = ft2/1.5;


ft = [ft1; ft2];

wo_len = zeros(1,simtime_len);    
zt = zeros(1,simtime_len);
zpt = zeros(1,simtime_len);
x0 = 0.5*randn(N,1);
z0 = 0.5*randn(1,1);

x = x0; 
r = tanh(x);
z = z0; 

figure;
P = (1.0/alpha)*eye(nRec2Out);
for q=1:N_patterns
    ti = 0;
for t = simtime
    ti = ti+1;	
    
    if mod(ti, nsecs/2) == 0
	subplot 211;
	plot(simtime, ft(q, :), 'linewidth', linewidth, 'color', 'green');
	hold on;
	plot(simtime, zt, 'linewidth', linewidth, 'color', 'red');
	title('training', 'fontsize', fontsize, 'fontweight', fontweight);
	legend('f', 'z');	
	xlabel('time', 'fontsize', fontsize, 'fontweight', fontweight);
	ylabel('f and z', 'fontsize', fontsize, 'fontweight', fontweight);
	hold off;
	
	subplot 212;
	plot(simtime, wo_len, 'linewidth', linewidth);
	xlabel('time', 'fontsize', fontsize, 'fontweight', fontweight);
	ylabel('|w|', 'fontsize', fontsize, 'fontweight', fontweight);
	legend('|w|');
	pause(0.5);	
    end
    
    % sim, so x(t) and r(t) are created.
    x = (1.0-dt)*x + M*(r*dt) + wf*(z*dt);
    r = tanh(x);
    z = wo(:, q)'*r;
    
    if mod(ti, learn_every) == 0
	% update inverse correlation matrix
	k = P*r;
	rPr = r'*k;
	c = 1.0/(1.0 + rPr);
	P = P - k*(k'*c);
    
	% update the error for the linear readout
	e = z-ft(q, ti);
	
	% update the output weights
	dw = -e*k*c;	
	wo(:,q) = wo(:, q) + dw;
    end
    
    % Store the output of the system.
    zt(ti) = z;
    wo_len(ti) = sqrt(wo(:, q)'*wo(:, q));	
end
error_avg = sum(abs(zt-ft(q, :)))/simtime_len;
disp(['Training MAE: ' num2str(error_avg,3)]);
end
disp(['Now testing... please wait.']);    

%% Testing %%

while (chosen < 1) || (chosen > N_patterns)
    chosen = input('Enter a number: ');
end

ti = 0;

for t = simtime				% don't want to subtract time in indices
    ti = ti+1;

    % sim, so x(t) and r(t) are created
    x = (1.0-dt)*x + M*(r*dt) + wf*(z*dt);
    r = tanh(x);
    z = wo(:, chosen)'*r;
    
    zpt(ti) = z;
end
error_avg = sum(abs(zpt-ft(chosen, :)))/simtime_len;
disp(['Testing MAE: ' num2str(error_avg,3)]);


% Compute the FFT
N_ = length(zpt(chosen, :)); % Number of points in the signal
Fs = 1/(simtime(2)-simtime(1)); % Sampling frequency
Y = fft(ft(chosen, :)); % Compute the FFT
P2 = abs(Y/N_); % Two-sided spectrum
P1 = P2(1:N_/2+1); % Single-sided spectrum
P1(2:end-1) = 2*P1(2:end-1);

% Compute the frequency vector
f = Fs*(0:(N_/2))/N_;

% Find the dominant frequency
[~, idx] = max(P1);
dominantFrequency = f(idx);

% Find the maximum and minimum amplitudes
maxAmplitude = max(ft(chosen, :));
minAmplitude = min(ft(chosen, :));

disp(['Desired Output Signal']);
disp(['min: ' num2str(minAmplitude, 3)]);
disp(['max:' num2str(maxAmplitude, 3)]);
disp(['dom frequ: ' num2str(dominantFrequency,5)]);

% Compute the FFT
Y = fft(zpt); % Compute the FFT
P2 = abs(Y/N_); % Two-sided spectrum
P1 = P2(1:N_/2+1); % Single-sided spectrum
P1(2:end-1) = 2*P1(2:end-1);

% Compute the frequency vector
f = Fs*(0:(N_/2))/N_;

% Find the dominant frequency
[~, idx] = max(P1);
dominantFrequency = f(idx);

% Find the maximum and minimum amplitudes
maxAmplitude = max(zpt);
minAmplitude = min(zpt);

disp(['Actual Output Signal']);
disp(['min: ' num2str(minAmplitude, 3)]);
disp(['max:' num2str(maxAmplitude, 3)]);
disp(['dom frequ: ' num2str(dominantFrequency,5)]);

figure;
hold on;
plot(simtime2, ft(chosen, :), 'linewidth', linewidth, 'color', 'green'); 
axis tight;
plot(simtime2, zpt, 'linewidth', linewidth, 'color', 'red');
axis tight;
title('simulation', 'fontsize', fontsize, 'fontweight', fontweight);
xlabel('time', 'fontsize', fontsize, 'fontweight', fontweight);
ylabel('f and z', 'fontsize', fontsize, 'fontweight', fontweight);
legend('f', 'z');

%%%	

