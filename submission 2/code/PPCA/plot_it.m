function plot_it(Y,C,ss,plo); 
clf;
    if plo==1
        plot(Y(:,1),Y(:,2),'.');
        hold on; 
        h=plot(C(1,1)*[-1 1]*(1+sqrt(ss)), (1+sqrt(ss))*C(2,1)*[-1 1],'r');
        h2=plot(0,0,'ro');
        set(h,'LineWi',4);
        set(h2,'MarkerS',10);set(h2,'MarkerF',[1,0,0]);
        axis equal;
    elseif plo==2
        len = 28;nc=1;
        colormap([0:255]'*ones(1,3)/255);
        d = size(C,2);
        m = ceil(sqrt(d)); n = ceil(d/m);
        for i=1:d; 
            subplot(m,n,i); 
            im = reshape(C(:,i),len,size(Y,2)/len,nc);
            im = (im - min(C(:,i)))/(max(C(:,i))-min(C(:,i))); 
            imagesc(im);axis off;
        end; 
    end
    drawnow;
    
    
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%