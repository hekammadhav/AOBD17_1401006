%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% original code by Jakob Verbeek

function [C, ss, M, X,Ye] = ppca_mv(Ye,d,dia,plo);
%
% implements probabilistic PCA for data with missing values, 
% using a factorizing distrib. over hidden states and hidden observations.
%
%  - The entries in Ye that equal NaN are assumed to be missing. - 
%
% [C, ss, M, X, Ye ] = ppca_mv(Y,d,dia,plo);
%
% Y   (N by D)  N data vectors
% d   (scalar)  dimension of latent space
% dia (binary)  if 1: printf objective each step
% plo (binary)  if 1: plot first PCA direction each step. 
%               if 2: plot eigenimages
%
% ss  (scalar)  isotropic variance outside subspace
% C   (D by d)  C*C' +I*ss is covariance model, C has scaled principal directions as cols.
% M   (D by 1)  data mean
% X   (N by d)  expected states
% Ye  (N by D)  expected complete observations (interesting if some data is missing)
%
% J.J. Verbeek, 2002. http://www.science.uva.nl/~jverbeek
%

%threshold = 1e-3;     % minimal relative change in objective funciton to continue
threshold = 1e-5;  

if plo; set(gcf,'Double','on'); end

[N,D] = size(Ye);
    
Obs   = ~isnan(Ye);
hidden = find(~Obs);
missing = length(hidden);

% compute data mean and center data
if missing
  for i=1:D;  M(i) = mean(Ye(find(Obs(:,i)),i)); end;
else
    M = mean(Ye);
end
Ye = Ye - repmat(M,N,1);

if missing;   Ye(hidden)=0;end

r     = randperm(N); 
C     = Ye(r(1:d),:)';     % =======     Initialization    ======
C     = randn(size(C));
CtC   = C'*C;
X     = Ye * C * inv(CtC);
recon = X*C'; 
recon(hidden) = 0;
ss    = sum(sum((recon-Ye).^2)) / ( (N*D)-missing);

count = 1; 
old   = Inf;


while count          %  ============ EM iterations  ==========
    
    if plo; plot_it(Ye,C,ss,plo);    end
   
    Sx = inv( eye(d) + CtC/ss );    % ====== E-step, (co)variances   =====
    ss_old = ss;
    if missing; proj = X*C'; Ye(hidden) = proj(hidden); end  
    X = Ye*C*Sx/ss;          % ==== E step: expected values  ==== 
    
    SumXtX = X'*X;                              % ======= M-step =====
    C      = (Ye'*X)  / (SumXtX + N*Sx );    
    CtC    = C'*C;
    ss     = ( sum(sum( (C*X'-Ye').^2 )) + N*sum(sum(CtC.*Sx)) + missing*ss_old ) /(N*D); 
    
    objective = N*(D*log(ss) +trace(Sx)-log(det(Sx)) ) +trace(SumXtX) -missing*log(ss_old);           
    rel_ch    = abs( 1 - objective / old );
    old       = objective;
    
    count = count + 1;
    if ( rel_ch < threshold) & (count > 5); count = 0;end
    if dia; fprintf('Objective: M %s    relative change: %s \n',objective, rel_ch ); end
    
end             %  ============ EM iterations  ==========


C = orth(C);
[vecs,vals] = eig(cov(Ye*C));
[vals,ord] = sort(diag(vals));
ord = flipud(ord);
vecs = vecs(:,ord);

C = C*vecs;
X = Ye*C;
 
% add data mean to expected complete data
Ye = Ye + repmat(M,N,1);


% ====  END === 