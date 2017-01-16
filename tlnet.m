% Threshold Linear Network a la Morrison & Curto et al
% (Diversity of emergent dynamics in competitive threshold-linear networks: 
% a preliminary report) https://arxiv.org/pdf/1605.04463v1.pdf
%
% dxi/dt=-xi+[sum_j Wij xj + theta]_+, theta=1
%
% Coded by Peter Thomas, CWRU, 2016-10-20
%
% Inspired by a talk and conversation at an MBI workshop.

%% choose example

%example='3_state_limit_cycle'; disp(example)
example='two_3_state_limit_cycles'; disp(example)
init='1'; % see init choices below
savedata='false'; % set to true to save data to ascii files.

%% general parameters
dt=.01;
tmax=10000; % long run
%tmax=10000; % even longer run
t=0:dt:tmax;
nt=length(t)-1;
delta=.5;
epsi=.25;
theta=1;
%alpha=1; % fully coupled -- one limit cycle wins
%alpha=.702; % coupling between limit cycles -- gives antisynchrony
%alpha=.703; % only one survives from IC [.5;0;0;.51;0;0]
%alpha=.7025; % border case -- looks like one will dominate eventually?
%alpha=.7; % antisynchronous
alpha=.01; % let's call this "weak coupling" -- phases separate gradually (by 4e3).
%alpha=.01; % let's call this "weak coupling" -- phases separate gradually (by 4e4).

%% Specify example

switch example
    case '3_state_limit_cycle'
        A=[0 0 1; 1 0 0; 0 1 0];
        nvtx=3;
        x=nan(nvtx,nt+1);
        x(:,1)=[.5;0;0];
        mask=ones(3);
    case 'two_3_state_limit_cycles'
        A1=[0 0 1; 1 0 0; 0 1 0];
        A=[A1,zeros(3);zeros(3),A1];
        nvtx=6;
        x=nan(nvtx,nt+1);
        switch init
            case '1'
                x(:,1)=[.6706567;.1601629;.1169525;
                        .09090289;.6662061;.2010911]; % phase shifted by almost half period (4/11.5)
            case '2'
                x(:,1)=[.6706567;.1601629;.1169525;
                        .6028187;.3250601;.04218902]; % oscillator 2 slightly ahead of oscillator 1
            case '3'
                x(:,1)=[.6028187;.3250601;.04218902;
                        .6706567;.1601629;.1169525]; % oscillator 1 slightly ahead of oscillator 2
            case '4'
                x(:,1)=[.5;0;0;.65;0;0]; % different
            case '5'
                x(:,1)=[.5;0;0;.51;0;0]; % slightly different
            case '6'
                x(:,1)=[.5;0;0;.5;0;0]; % identical -- see unstable synchronous limit cycle, with alpha=0.1 (maybe other values too)
        end
        mask=[ones(3),alpha*ones(3);alpha*ones(3),ones(3)];
end

%% Connections get weight -1+epsi, otherwise -1-delta.
% Diagonal elements are zero.
% If necessary, use mask to remove some connections entirely.
W=(A==1)*(-1+epsi)+(A==0)*(-1-delta); 
W=W-diag(diag(W)); 
W=W.*mask;

%% lookup tables for limit cycles
x1 = load('x1.dat');
x2 = load('x2.dat');
x3 = load('x3.dat');
x1_size=length(x1);

%% initial phase values:
% phase 1
[~,I] = min((x(1,1)-x1(:,2)).^2 + (x(2,1)-x2(:,2)).^2 + (x(3,1)-x3(:,2)).^2);
phase1(1)=I/x1_size;

% phase 2
[~,I] = min((x(4,1)-x1(:,2)).^2 + (x(5,1)-x2(:,2)).^2 + (x(6,1)-x3(:,2)).^2);
phase2(1)=I/x1_size;

%phase = zeros(nt);

%% main loop
for jt=1:nt
    x(:,jt+1)=(1-dt)*x(:,jt)+dt*max(0,theta+W*x(:,jt));
    
    % find approximate phase 1 and record value
    [~,I] = min((x(1,jt)-x1(:,2)).^2 + (x(2,jt)-x2(:,2)).^2 + (x(3,jt)-x3(:,2)).^2);
    phase1(jt+1)=I/x1_size;
    
    % find approx phase 2 and record value
    [~,I] = min((x(4,jt)-x1(:,2)).^2 + (x(5,jt)-x2(:,2)).^2 + (x(6,jt)-x3(:,2)).^2);
    phase2(jt+1)=I/x1_size;
    
end

%% phase diff (theory) loop

% load phase diff RHS lookup table
data = load('tlnet-2hodd.dat');

dx = data(2,1)-data(1,1);

xq = data(1,1):.01:data(end,1);

per = data(end,1);

phi(1) = (phase2(1) - phase1(1))*per;

% make into usable function
phsrhs = griddedInterpolant(data(:,1),data(:,2),'nearest');

%for jt=1:nt
%    phi(jt+1) = phi(jt) + alpha*dt*phsrhs(mod(per*phi(jt),per))/per;
%end
for jt=1:nt
    phi(jt+1) = phi(jt) + alpha*dt*phsrhs(mod(phi(jt),per));
end

phi = phi/per;

%% plot output
figure(1),clf
subplot(4,1,1)
initial = x(:,1:40/dt);
plot(t(1:40/dt),initial,'LineWidth',3)
xlim([0 40])
ylim([0 theta])
set(gca,'FontSize',20)
title('Morrison-Curto Threshold Linear Network','FontSize',20)
switch example
    case '3_state_limit_cycle'
        legend('1','2','3')
    case 'two_3_state_limit_cycles'
        legend('1','2','3','4','5','6')
end

subplot(4,1,2)

plot(t,x,'LineWidth',3)
set(gca,'FontSize',20)
ylim([0 theta])

subplot(4,1,3)
final = x(:,end-40/dt:end);
plot(t(end-40/dt:end),final,'LineWidth',3)
xlim(tmax+[-40 0])
ylim([0 theta])
set(gca,'FontSize',20)

subplot(4,1,4)
numerics = mod(phase2-phase1+.5,1)-.5;
scatter(t,numerics,25,'black','filled');hold on;
plot(t,phi,'green','LineWidth',3)
%xlim(tmax+[-40 0])
ylim([-.5 .5])
set(gca,'FontSize',20)

% save data files for figure generation in python
if strcmp(example,'two_3_state_limit_cycles') && strcmp(savedata,'true');
    f1 = strcat(strcat('tlnetphase_theory',init),'.txt');
    f2 = strcat(strcat('tlnet_phase_and_full_t',init),'.txt');
    f3 = strcat(strcat('tlnetphase_full',init),'.txt');
    f4 = strcat(strcat('tlnet_full_sol',init),'.txt');
    
    save(f1,phi,'-ascii')
    save(f2,t,'-ascii')
    save(f3,numerics,'-ascii')
    save(f4,x,'-ascii')
end

shg


