function [pc,W,data_mean,xr,evals,percentVar]=ppca(data,k)
% PCA applicable to 
%   - extreme high-dimensional data (e.g., gene expression data) and
%   - incomplete data (missing data)
%
% probabilistic PCA (PPCA) [Verbeek 2002]
% based on sensible principal components analysis [S. Roweis 1997]
%  code slightly adapted by M.Scholz
%
% pc = ppca(data)
% [pc,W,data_mean,xr,evals,percentVar]=ppca(data,k)
%
%  data - inclomplete data set, d x n - matrix
%          rows:    d variables (genes or metabolites)
%          columns: n samples
%
%  k  - number of principal components (default k=2)
%  pc - principal component scores  (feature space)
%       plot(pc(1,:),pc(2,:),'.')
%  W  - loadings (weights)
%  xr - reconstructed complete data matrix (for k components)
%  evals - eigenvalues
%  percentVar - Variance of each PC in percent
%
%    pc=W*data
%    data_recon = (pinv(W)*pc)+repmat(data_mean,1,size(data,2))
%
% Example:
%    [pc,W,data_mean,xr,evals,percentVar]=ppca(data,2)
%    plot(pc(1,:),pc(2,:),'.'); 
%    xlabel(['PC 1 (',num2str(round(percentVar(1)*10)/10),'%)',]);
%    ylabel(['PC 2 (',num2str(round(percentVar(2)*10)/10),'%)',]);
%
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin==1
  k=2
end
 
  [C,ss,M,X,Ye]=ppca_mv(data',k,0,0);
  xr=Ye';
  W=C';
  data_mean=M';
  pc=X';
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% calculate variance of PCs
 
  for i=1:size(data,1)  % total variance, by using all available values
   v(i)=var(data(i,~isnan(data(i,:)))); 
  end
  total_variance=sum(v(~isnan(v)));
  
  evals=nan(1,k);
  for i=1:k 
    data_recon = (pinv(W(i,:))*pc(i,:)); % without mean correction (does not change the variance)
    evals(i)=sum(var(data_recon'));
  end
  
  percentVar=evals./total_variance*100;
  
%    cumsumVarPC=nan(1,k);  
%   for i=1:k 
%     data_recon = (pinv(W(1:i,:))*pc(1:i,:)) + repmat(data_mean,1,size(data,2));
%     cumsumVarPC(i)=sum(var(data_recon')); 
%   end
%   cumsumVarPC
  