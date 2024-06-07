disp('Clearing workspace.');
clear;

linewidth = 3;
fontsize = 14;
fontweight = 'bold';

N = 1000;
p = 0.1;                % 10% sparsity
g = 1.5;				% g greater than 1 leads to chaotic networks.
alpha = 1.0;
nsecs = 5000; %3000; %2000; %1440;
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

error = zeros(101, 1);
percentage = 0:0.1:10;
train_error = zeros(101, 1);


for j=0:100
int_error = zeros(5, 1);
int_train_error = zeros(5, 1);

for s=1:5

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

figure;
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
int_train_error(s) = error_avg;

% Now test. 
ti = 0;


%%% Connections killing
nonZeroIndices = find(M);
selectedIndices = nonZeroIndices(randperm(length(nonZeroIndices), j));
M(selectedIndices) = 0;


for t = simtime				% don't want to subtract time in indices
    ti = ti+1;
    % sim, so x(t) and r(t) are created
    x = (1.0-dt)*x + M*(r*dt) + wf*(z*dt);
    r = tanh(x);
    z = wo'*r;
    
    zpt(ti) = z;
end
error_avg = sum(abs(zpt-ft2))/simtime_len;
disp(['Testing MAE: ' num2str(error_avg,3)]);

int_error(s) = error_avg;
end
disp(['j =' num2str(j)]);
train_error(j+1) = 1-mean(int_train_error);
error(j+1) = 1-mean(int_error);
end

figure;
subplot 211;
plot(percentage, train_error, 'linewidth', linewidth, 'color', 'green');
hold on;
plot(percentage, error, 'linewidth', linewidth, 'color', 'red');
title('Pattern similarity vs destroyed connections', 'fontsize', fontsize-5, 'fontweight', fontweight);
xlabel('Destoryed connection percentage', 'fontsize', fontsize-5, 'fontweight', fontweight);
hold on;
ylabel('Pattern similarity', 'fontsize', fontsize-5, 'fontweight', fontweight);
legend('Training similarity', 'testing similarity');