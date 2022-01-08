function [model,log_like,model_perf,Q_rew,Q_puff,P,P_rew,P_puff,AIC,entropy,exitflag] = UpgradedRW_CBC_v4(choices,rewards,puffs,param_included,operation,params)
%Upgraded Rescorla Wagner Model fitter, uses different alpha weights
%   depending on task outcomes
%   INPUT: choices: vector, 0s for left, 1s for right, nan/anything else
%       for timeout
%           rewards: vector, 0s for no reward, 1 for reward
%           puffs: vector, 0s for no reward, 1 for reward
%           param_included: 10-element vector: 1 if factor included; 0 if
%             not
%           operation: 'fit' or 'eval'
%           params: if 'eval' provide model parameters for calculating Q
%           and P
%   OUTPUT: model: vector of model parameters:
%               [a_rew,a_unrew,a_forget,beta_reward,beta_bias,a_puff,a_unpuff,a_forget_puff,beta_puff,beta_repbias]
%           log_like: log(L) for optimal RW model
%           model_perf: fraction correct choices by model
%           Qrew: array of reward action values, 1st row for right, 2nd row for left
%               side
%           Qpuff: array of puff action values, 1st row for right, 2nd row for left
%               side
%           P: array of probability values per trial, 1st row for right,
%               2nd row for left
%           Prew: same as P, but ignoring puff Q-values
%           Ppuff: same as P, but ignoring reward Q-values
%           AIC: aikaike information criterion (2k - ln(L))
%           entropy: cross entropy of model over given data
%           exitflag: fminsearchbnd exitflag



%% Set parameters
%parameter order: a_rew,a_unrew,a_forget,beta,bias,a_puff,a_unpuff,a_forget_puff,beta_puff,choice_bias]
x0 = [0.4,0.4,0.4,5,0.0,0.4,0.4,0.4,5,0]; % initial values
x1 = [0,0,0,0.1,-1,0,0,0,0.1,-1]; % lower bound
x2 = [1,1,1,10,1,1,1,1,10,1]; % higher bound


% set parameters to fixed value if not included in the model
x0([1 2 3 5 6 7 8 10]) = x0([1 2 3 5 6 7 8 10]) .* param_included([1 2 3 5 6 7 8 10]);
x1([1 2 3 5 6 7 8 10]) = x1([1 2 3 5 6 7 8 10]) .* param_included([1 2 3 5 6 7 8 10]);
x2([1 2 3 5 6 7 8 10]) = x2([1 2 3 5 6 7 8 10]) .* param_included([1 2 3 5 6 7 8 10]);

if param_included(4) == 0
    x0(4) = 1;
    x1(4) = 1;
    x2(4) = 1;
end
if param_included(9) == 0
    x0(9) = 1;
    x1(9) = 1;
    x2(9) = 1;
end


%% Fit model
options = optimset('MaxFunEvals',100000000);
options = optimset('MaxIter',1000000);

if operation == "fit"
    [model,log_like,exitflag] = fminsearchbnd(@AI,x0,x1,x2,options);  %finds maximum likelihood parameters
elseif operation == "eval"
    model = params;     
    log_like = AI(model);
    exitflag = 1;       
end



%% Evaluate model and form predictions
log_like = -log_like;   %calculation uses negative likelihood to make search for minimum instead of maximum
model_perf = evaluate(choices,rewards,puffs,model,param_included);

Q_rew = zeros([2,length(choices)]);
Q_puff = zeros([2,length(choices)]);
[Q_rew_r,Q_rew_l,Q_puff_r,Q_puff_l] = actionVal(choices,rewards,puffs,model(1),model(2),model(3),model(6),model(7),model(8),param_included);

Q_rew(1,:) = Q_rew_r;
Q_rew(2,:) = Q_rew_l;    %saves right side action values in first row, left side action values in 2nd row
Q_puff(1,:) = Q_puff_r;
Q_puff(2,:) = Q_puff_l;    %saves right side action values in first row, left side action values in 2nd row

P = zeros([2,length(choices)]);
for i=1:length(choices)
    P(1,i) = softmax(Q_rew_r(i),Q_rew_l(i),Q_puff_r(i),Q_puff_l(i),model(4),model(5),model(9),model(10),param_included,choices);
    P(2,i) = 1- P(1,i);
end

% calculate Q and P for reward or puff only
P_rew = zeros([2,length(choices)]);
P_puff = zeros([2,length(choices)]);
for i=1:length(choices)
    P_rew(1,i) = softmax(Q_rew_r(i),Q_rew_l(i),0,0,model(4),model(5),model(9),model(10),param_included,choices);
    P_rew(2,i) = 1- P_rew(1,i);
    P_puff(1,i) = softmax(0,0,Q_puff_r(i),Q_puff_l(i),model(4),model(5),model(9),model(10),param_included,choices);
    P_puff(2,i) = 1- P_puff(1,i);
end

n_parameters = sum(param_included);
AIC = 2 * n_parameters - 2 * log_like;
entropy = crossE(P(1,:),choices);       %%% ADDED IN TO CALCULATE CROSS ENTROPY AS WELL


%% Function to be fitted
    function log_like = AI(parameters)
        %find the entropy of an AI given certain parameters
        alpha_rew = parameters(1);
        alpha_unrew = parameters(2);
        gamma_rew = parameters(3);      %forgetting rate, calculated as (1-gamma) * Q
        beta = parameters(4);
        bias = parameters(5);
        
        alpha_puff = parameters(6);
        alpha_unpuff = parameters(7);
        gamma_puff = parameters(8);      %forgetting rate, calculated as (1-gamma) * Q
        beta_puff = parameters(9);
        choice_bias = parameters(10);
        
        numTrials = length(choices);
        
        [Q_rewr,Q_rewl,Q_puffr,Q_puffl] = actionVal(choices,rewards,puffs,alpha_rew,alpha_unrew,gamma_rew,alpha_puff,alpha_unpuff,gamma_puff,param_included);
        
        %calculate probability for each trial based on softmax equation
        prob_right = zeros([1,numTrials]);
        for trial=1:numTrials
            prob_right(trial) = softmax(Q_rewr(trial),Q_rewl(trial),Q_puffr(trial),Q_puffl(trial),beta,bias,beta_puff,choice_bias,param_included,choices);
        end
        
        log_like = 0;
        for trial=1:numTrials
            if choices(trial) == 1
                log_like = log_like + log(prob_right(trial));
            elseif choices(trial) == 0
                log_like = log_like + log(1-prob_right(trial));
            end
        end
        log_like = -log_like;  %changed to negative to allow search for minimum
    end
end


function prob = softmax(Q1_rew,Q2_rew,Q1_puff,Q2_puff,beta,bias,beta_puff,choice_bias,param_included,choices)
%softmax equation to determine probability of Q1 action value side
%     prob = 1/(1+exp(bias - beta*(Q1_rew-Q2_rew)));
Q_diff_rew = beta *(Q1_rew - Q2_rew);
Q_diff_puff = beta_puff * (Q1_puff - Q2_puff);

choice_bias_trial = zeros(size(Q1_rew));
for idx = 1:length(choice_bias_trial)
    if idx ~= 1 && choices(idx-1) == 1
        choice_bias_trial(idx) = choice_bias;
    elseif idx ~= 1 && choices(idx-1) == 0
        choice_bias_trial(idx) = - choice_bias;
    end
end
prob = 1/(1+exp(bias - Q_diff_rew + Q_diff_puff - choice_bias_trial));
end


function [Q_rew_r,Q_rew_l,Q_puff_r,Q_puff_l] = actionVal(choices,rewards,puffs,a_rew,a_unrew,gamma_rew,a_puff,a_unpuff,gamma_puff,param_included)
%calculate the action values using defined choices and rewards of mouse
%task, calculate with given parameters

if param_included(2) == 0
    a_unrew = a_rew;
end
if param_included(7) == 0
    a_unpuff = a_puff;
end
if param_included(3) == 0
    gamma_rew = a_unrew;
end
if param_included(8) == 0
    gamma_puff = a_unpuff;
end

numTrials = length(choices);
Q_rew_r = zeros([1,numTrials]);
Q_rew_l = zeros([1,numTrials]);  %initialize to 0 for starting action values
Q_puff_r = zeros([1,numTrials]);
Q_puff_l = zeros([1,numTrials]);  %initialize to 0 for starting action values
for trial=2:numTrials
    outcome_rew = rewards(trial-1);
    if outcome_rew == 1
        alpha_rew = a_rew;
    else
        alpha_rew = a_unrew;
    end
    outcome_puff = puffs(trial-1);
    if outcome_puff == 1
        alpha_puff = a_puff;
    else
        alpha_puff = a_unpuff;
    end
    
    
    if choices(trial-1) == 0
        Q_rew_l(trial) = Q_rew_l(trial-1) + alpha_rew * (outcome_rew - Q_rew_l(trial-1));  %left action value update
        Q_rew_r(trial) = Q_rew_r(trial-1) * (1-gamma_rew);
        Q_puff_l(trial) = Q_puff_l(trial-1) + alpha_puff * (outcome_puff - Q_puff_l(trial-1));  %left action value update
        Q_puff_r(trial) = Q_puff_r(trial-1) * (1-gamma_puff);
    elseif choices(trial-1) == 1
        Q_rew_r(trial) = Q_rew_r(trial-1) + alpha_rew * (outcome_rew - Q_rew_r(trial-1));    %right action value update
        Q_rew_l(trial) = Q_rew_l(trial-1) * (1-gamma_rew);
        Q_puff_r(trial) = Q_puff_r(trial-1) + alpha_puff * (outcome_puff - Q_puff_r(trial-1));    %right action value update
        Q_puff_l(trial) = Q_puff_l(trial-1) * (1-gamma_puff);
    else
        Q_rew_r(trial) = Q_rew_r(trial-1) * (1-gamma_rew);  %timeout case, both action values decay
        Q_rew_l(trial) = Q_rew_l(trial-1) * (1-gamma_rew);
        Q_puff_r(trial) = Q_puff_r(trial-1) * (1-gamma_puff);  %timeout case, both action values decay
        Q_puff_l(trial) = Q_puff_l(trial-1) * (1-gamma_puff);
    end
    
end
end


function fraction_correct = evaluate(choices,rewards,puffs,parameters,param_included)
%calculates goodness of fit, based on how many times the rescorla wagner
%model generates the correct decision, based on whether generated
%probability is greater than 50%
numTrials = length(choices);
a_rew = parameters(1);
a_unrew = parameters(2);
gamma_rew = parameters(3);
beta = parameters(4);
bias = parameters(5);
a_puff = parameters(6);
a_unpuff = parameters(7);
gamma_puff = parameters(8);      %forgetting rate, calculated as (1-gamma) * Q
beta_puff = parameters(9);
choice_bias = parameters(10);

[Q_rew_r,Q_rew_l,Q_puff_r,Q_puff_l] = actionVal(choices,rewards,puffs,a_rew,a_unrew,gamma_rew,a_puff,a_unpuff,gamma_puff,param_included);
p_r = zeros([1,numTrials]);
for trial=1:numTrials
    p_r(trial) = softmax(Q_rew_r(trial),Q_rew_l(trial),Q_puff_r(trial),Q_puff_l(trial),beta,bias,beta_puff,choice_bias,param_included,choices);
end

correct = 0;    %number of times model generates correct answer
for trial=1:numTrials
    if choices(trial) == 1 && p_r(trial) > 0.5
        correct = correct + 1;
    elseif choices(trial) == 0 && p_r(trial) < 0.5
        correct = correct + 1;
    end
end
num_Choices = sum(~isnan(choices));
fraction_correct = correct/num_Choices;
end


function entropy = crossE(Pr,choices)
%calculate the entropy of the model
entropy = 0;
entropy = -1 * nanmean((1-choices) .* log(1-Pr) + choices .* log(Pr));
end
