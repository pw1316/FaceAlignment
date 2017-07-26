

function [ output_args ] = LBFTrain( T, NumTrees, TreeDepth )

global TRAIN_DATA_SIZE
global TRAIN_DATA_PATH
global TEST_DATA_PATH
global NUM_LANDMARKS
global NUM_AUG

TRAIN_DATA_SIZE=30;
NUM_LANDMARKS=74;
NUM_AUG=10;
TRAIN_DATA_PATH='F:/m/2014/lbf/data/ours/4_30_Xi_CCX1/';
TEST_DATA_PATH='F:/m/2014/lbf/data/ours/4_30_Xi_CCX1/';

    [Train_Images, Train_GTShapes, Train_S,mean_shape]=init();    
    Nsamples=size(Train_S,1);
    
    for t=1:T % stage
        fprintf('==================== Training stage %d ====================\n',t);
        
        train_fname=strcat(TRAIN_DATA_PATH,sprintf('lbf%d.mat',t));
        
        if exist(train_fname,'file')==2
            load(train_fname,'rforests','nleaves');
        else % train
            tree_ = struct('nodes',[],'root_id',-1);
            rforests = repmat(tree_,NUM_LANDMARKS,NumTrees);
            nleaves=zeros(NUM_LANDMARKS,1);
            
            for l=1:NUM_LANDMARKS
                fprintf('------ Building random forests for landmark %d ------\n',l);
        
                landmarks=Train_S(:,((2*l-1):2*l));
                GTlandmarks=Train_GTShapes(:,((2*l-1):2*l));

                tic();
                [rforests(l,:),nleaves(l)]=buildRFforLandmark(Train_Images,GTlandmarks,mean_shape,landmarks,NumTrees,TreeDepth);
                toc();
            end 
            
            %dump rfs, nleaves
            save(train_fname,'rforests','nleaves');
        end
        
        if(isempty(whos(matfile(train_fname),'W')))
            % TRAIN_DATA_SIZE = size(Train_GTShapes,1)
            GT_deltaS=Train_GTShapes(ceil((1:NUM_AUG*TRAIN_DATA_SIZE)/NUM_AUG),:)-Train_S;
            %GT_deltaS=kron(Train_GTShapes,ones(NUM_AUG,1))-Train_S;

            tot_leaves=sum(nleaves);
            W=zeros(tot_leaves,2*NUM_LANDMARKS);
            for coord_i=1:2*NUM_LANDMARKS
                
                disp_text=sprintf('Regressing W %d/%d ',coord_i,2*NUM_LANDMARKS);
                showProgress(disp_text,coord_i/(2*NUM_LANDMARKS));

                Phi=zeros(Nsamples,tot_leaves); % n * DL
                %Phi=[];
                for i=1:Nsamples
                    img=Train_Images(int32(ceil(i/NUM_AUG)),:,:);
                    img=reshape(img,size(img,2),size(img,3));

                    phi=[];
                    for l=1:NUM_LANDMARKS
                        landmark=Train_S(int32(ceil(i/NUM_AUG)),((2*l-1):2*l));

                        phi_l=zeros(1,nleaves(l));                   

                        [~,nonzero_leaves]=runRF(rforests(l,:),img,landmark);
                        phi_l(nonzero_leaves)=1;
                        phi=[phi,phi_l]; % concate local binary features
                    end
                    Phi(i,:)=phi;
                    %Phi=[Phi;sparse(phi)];
                end

                w=ridgeRegression( GT_deltaS(:,coord_i), Phi, 1.0);

                deltaS=Phi*w;
                Train_S(:,coord_i)=Train_S(:,coord_i)+deltaS;
                W(:,coord_i)=w;
            end
            save(train_fname,'W','-append');
        end
    end

end

%{
function [ Shape ] = LBFTest( img )
global TRAIN_DATA_PATH
global NUM_LANDMARKS

    Shape=mean_shape;
    for t=1:T % stage
                
        train_fname=strcat(TRAIN_DATA_PATH,sprintf('lbf%d.mat',t));
        if exist(train_fname,'file')~=2
            return;
        end
        load(train_fname,'rforests','nleaves','W');
        
        landmarks=Shape;
        for coord_i=1:2*NUM_LANDMARKS
            coord_i
            
            phi=[];
            for l=1:NUM_LANDMARKS
                landmark=landmarks(((2*l-1):2*l));

                phi_l=zeros(1,nleaves(l));                   

                [~,nonzero_leaves]=runRF(rforests(l,:),img,landmark);
                phi_l(nonzero_leaves)=1;
                phi=[phi,phi_l]; % concate local binary features
            end

            deltaS= phi*W(:,coord_i);
            Shape(coord_i)=Shape(coord_i)+deltaS;
        end
        
    end
    
    imshow(uint8(img));
    hold on;
    plot(Shape(1:2:end),Shape(2:2:end),'.');
end
%}

function done = clamp(low, value, high)
    done = min(high, max(low, value));
end

function [fea] = sampleFeatures(img,center,R,numP)
    fea=zeros(numP*(numP-1)/2,5);
    % non-uniform?
    r = sqrt(rand(numP,1))*R;
    theta = rand(numP,1)*(2*pi);
    dx =  r.*cos(theta);
    dy =  r.*sin(theta);
    
    x=center(1)+dx;
    y=center(2)+dy;
    
    % int16-int16
    sz=size(img);
    pts=int16(img(sub2ind(sz,clamp(1,round(y),sz(1)),clamp(1,round(x),sz(2)))));
    pxdiff=bsxfun(@minus,pts',pts);

    linidx=find(triu(ones(numP,numP),1)>0);
    
    % note the order
    [idx2,idx1]=ind2sub([numP,numP],linidx);
    
    fea(:,1)=dx(idx1);
    fea(:,2)=dy(idx1);
    fea(:,3)=dx(idx2);
    fea(:,4)=dy(idx2);
    fea(:,5)=pxdiff(linidx);
end

function [the_tree,node_id,leaf_id] =  buildTree(the_tree,landmarks,GTlandmarks,MFeatures,sampleIdxs,cur_d,D,leaf_id)
global NUM_AUG

    node=struct('left','null','right','null','value','null');
    
    Nsamples=length(sampleIdxs);
    if (cur_d==D||Nsamples==1)
        offset=GTlandmarks(int32(ceil(sampleIdxs/NUM_AUG)),:)-landmarks(sampleIdxs,:);
        node.value=[mean(offset,1),leaf_id];
        the_tree.nodes=[the_tree.nodes;node];
        
        node_id=length(the_tree.nodes);
        leaf_id=leaf_id+1;
        return;
    end
    
    M=size(MFeatures,2);
    m=500;

    mOutOfM=randsample(M,m); % select same feature as previous levels?
    mFeaCoords=MFeatures(sampleIdxs,mOutOfM,1:4);
	mFeatures=reshape(MFeatures(sampleIdxs,mOutOfM,5),Nsamples,m);

	[mFeatures,mFeaSortIdx]=sort(mFeatures); % sort samples for each feature

	splitVar=-1.0;
	split_m=-1; % split on which feature
    split_i=-1; % split on which sample(threshold)
    split_coords=zeros(4,1);
    split_threshold=0;
    
    mFea_sum=sum(mFeatures);
    mFea_sum2=sum(mFeatures.^2);
    invN=1.0/Nsamples;
    
    for j=1:m % try split on each feature

		maxReduce=-1.0;
		maxI=-1;

        tot_sum=mFea_sum(j);
        tot_sum2=mFea_sum2(j);
		tot_var=abs(invN*(tot_sum2-invN*tot_sum*tot_sum));
        
        % find threshold
		left_sum=0;
        left_sum2=0;
		for i=1:Nsamples
            feaval=mFeatures(i,j);
			left_sum=left_sum+feaval;
			left_sum2=left_sum2+feaval*feaval;

			invL=1.0/(i);
			invR=1.0/(Nsamples-i);

			right_sum=tot_sum-left_sum;
            right_sum2=tot_sum2-left_sum2;
			var1=abs(invL*(left_sum2-invL*left_sum*left_sum));
			var2=abs(invR*(right_sum2-invR*right_sum*right_sum));

			vreduce=tot_var-var1-var2;
			if (vreduce>maxReduce)
				maxReduce=vreduce;
				maxI=i;
            end
        end
        
        
		if(maxReduce>splitVar)
			splitVar=maxReduce;
			split_i=maxI;
			split_m=j;
            split_threshold=mFeatures(split_i,split_m);
            split_coords=reshape(mFeaCoords(split_i,split_m,:),4,1);
        end
        
    end

    idxL=mFeaSortIdx(1:split_i,split_m);
    idxR=mFeaSortIdx(split_i+1:end,split_m);
    
    %T1=MFeatures(idxL,:,:);
    %T2=MFeatures(idxR,:,:);
    %L1=landmarks(idxL,:);
    %L2=landmarks(idxR,:);

	[the_tree, left_id, leaf_id]=buildTree(the_tree,landmarks,GTlandmarks,MFeatures,idxL,cur_d+1,D,leaf_id);
	[the_tree, right_id, leaf_id]=buildTree(the_tree,landmarks,GTlandmarks,MFeatures,idxR,cur_d+1,D,leaf_id);
    
    % if internal
    node.left=left_id;
    node.right=right_id;
    node.value=[split_coords; split_threshold]; % feature is defined by the diff of two pixels, store the relative positions
    % shape-indexed feature, relative to last shape
    
    the_tree.nodes=[the_tree.nodes;node];
    node_id=length(the_tree.nodes);
    
end

function [offset,nonzero_leaves]=runRF(RF,img,landmark)
    Ntrees=length(RF);
    
    offset=zeros(1,2);
    nonzero_leaves=zeros(1,Ntrees);
    
    for ti=1:Ntrees
        tree_=RF(ti);
        node=tree_.nodes(tree_.root_id);
        %tree_D=...; % leaves count
        
        while(1)
        	if(length(node.value)==5)
                pt1=landmark+[node.value(1),node.value(2)];
                pt2=landmark+[node.value(3),node.value(4)];
                threshold=node.value(5);
                
                sz=size(img);
                col1=img(sub2ind(sz,clamp(1,round(pt1(2)),sz(1)),clamp(1,round(pt1(1)),sz(2)))); % y,x
                col2=img(sub2ind(sz,clamp(1,round(pt2(2)),sz(1)),clamp(1,round(pt2(1)),sz(2))));
                fea=int16(col1)-int16(col2);
                
                if(fea<=threshold)
                    node=tree_.nodes(node.left);
                else
                    node=tree_.nodes(node.right);
                end
            else
                offset=offset+node.value(:,1:2);
                nonzero_leaves(ti)=node.value(3);
                break;
            end
        end
    end
end

% [0,1]
function showProgress(txt,i)
    persistent nchar
    
    if(isempty(nchar))
        nchar=0;
    end
    
    fprintf(repmat('\b', 1, nchar));
    nchar=fprintf('%s %.2f%%',txt,i*100);

    if(i>=1)
        nchar=0;
        fprintf('\n');
    end
end

function [minRF,nleaves] = buildRFforLandmark(Train_Images,Train_GTLandmarks,mean_shape,landmarks,Ntrees,D)
global NUM_AUG
    Nsamples=size(landmarks,1);
    Npixels=400;
    Nfeatures=Npixels*(Npixels-1)/2;
   
    pupil_d=120; %mean_shape(1);
    R=sqrt(2).^(9:-1:0);
    R=R*(pupil_d/sqrt(2)/R(1)); % largest radius set to sqrt(2)*pupil_distance
    
    min_oob_error=inf;
    
	
	for ri=1:length(R) % cv on radius
		r=R(ri);
        
        fprintf('****** radius = %f ******\n',r);

        % generate features
        MFeatures=zeros(Nsamples,Nfeatures,5);
        for i=1:Nsamples
            
            showProgress('sampling features',i/Nsamples);
            
            img=Train_Images(int32(ceil(i/NUM_AUG)),:,:);
            img=reshape(img,size(img,2),size(img,3));
            
            MFeatures(i,:,:)=sampleFeatures(img,landmarks(i,:),r,Npixels);
            
        end

        
        Ts_idx=zeros(Ntrees,Nsamples);
        % build forest
        tree = struct('nodes',[],'root_id',-1);
        RF = repmat(tree,Ntrees,1);
        leaf_id = 1;
		for nt=1:Ntrees
            showProgress('building tree',nt/Ntrees);
            
            % sample with replacement
            Ts_idx(nt,:)=randi(Nsamples,1,Nsamples);
            tree.nodes=[];
            [tree,root_id,leaf_id]=buildTree(tree,landmarks,Train_GTLandmarks,MFeatures,Ts_idx(nt,:),1,D,leaf_id);
            tree.root_id=root_id;
            RF(nt)=tree;
            Ts_idx(nt,Ts_idx(nt,:))=-1;
        end
        
        % oob error
        %for nt=1:Ntrees
        %    oob_idx=setdiff(1:Nsamples,Ts_idx(nt));
        %end

        oob_error=0;
		for i=1:Nsamples
            showProgress('computing oob error',i/Nsamples);
            
            img=Train_Images(int32(ceil(i/NUM_AUG)),:,:);
            img=reshape(img,size(img,2),size(img,3));
            gt_landmark=Train_GTLandmarks(int32(ceil(i/NUM_AUG)),:); % ground-truth landmark
            landmark=landmarks(i,:);
            
			offset=zeros(1,2);
			for nt=1:Ntrees
				if(Ts_idx(nt,i)==-1) % included in the tree's traning samples
                    continue;
                end
                [delta,~]=runRF(RF(nt),img,landmark); % run single tree
                offset=offset+delta; % todo: average?
            end
            
            gt_delta=gt_landmark-landmark; % ground-truth delta
            error2=gt_delta-offset;
            oob_error=oob_error+abs(error2*error2'); % todo: sqrt?
        end
        
        % keep the tree with minimal error
        if(oob_error<min_oob_error)
            minRF=RF; % ref to the whole tree??
            min_oob_error=oob_error;
            nleaves=leaf_id-1;
        end
        
    end
end

function w = ridgeRegression(y, X, lambda)
    P=size(X,2); %??
    w=pinv(X'*X+lambda*eye(P))*X'*y;
end

function img=loadPNG(fpath)
    img=rgb2gray(imread(fpath));
end

function shp=loadShape(fpath,width,height)
global NUM_LANDMARKS
    fid=fopen(fpath,'rt');
    fscanf(fid,'%d',1);    
    shp=fscanf(fid,'%f',2*NUM_LANDMARKS);
    fclose(fid);
    
    shp(1:2:end)=shp(1:2:end)*width;
    shp(2:2:end)=(1-shp(2:2:end))*height;
end

function [width,height]=getImageSize()
global TRAIN_DATA_PATH

    fpath=strcat(TRAIN_DATA_PATH,'0000.png');
    t=imread(fpath);
    width=size(t,2);
    height=size(t,1);
end

%{
function img=loadBioIDImage(fpath)
    fid=fopen(fpath,'rb');
    fscanf(fid,'%c',2);
    Dim=fscanf(fid,'%d',3);
    width=Dim(1);height=Dim(2);
    fseek(fid,1,0);
    img=fread(fid,[width,height],'uint8=>int16');
    img=img';
    %img=fread(fid,width*height,'uint8=>int16');
    %img = reshape(img, [1 width height]);
    %img = permute(img, [3 2 1]);
    fclose(fid);
end

function shp=loadBioIDShape(fpath)
global NUM_LANDMARKS
    fid=fopen(fpath,'rt');
    fgetl(fid);fgetl(fid);fgetl(fid);
    
    shp=fscanf(fid,'%f',2*NUM_LANDMARKS);
    fclose(fid);
end

function [width,height]=getBioIDImageSize()
global TRAIN_DATA_PATH

    fpath=strcat(TRAIN_DATA_PATH,'bioid_0000.pgm');
    fid=fopen(fpath,'rb');
    fscanf(fid,'%c',2);
    A=fscanf(fid,'%d',2);
    width=A(1);height=A(2);
    fclose(fid);
end
%}

% Train_S stores the initial shape, the corresponding image and gt_shape is
% idx=int32(fix((i+NUM_AUG-1)/NUM_AUG)); or idx=int32(ceil(i/NUM_AUG));
% Train_Images(:,:,idx) and Train_GTShapes(:,idx)
function [Train_Images, Train_GTShapes, Train_S, mean_shape]=init()
global TRAIN_DATA_SIZE
global TRAIN_DATA_PATH
global NUM_LANDMARKS
global NUM_AUG

    [width,height]=getImageSize();
    Train_Images=zeros(TRAIN_DATA_SIZE,height,width,'uint8');
    Train_GTShapes=zeros(TRAIN_DATA_SIZE,2*NUM_LANDMARKS);

    for i=0:TRAIN_DATA_SIZE-1
        
        % bioid
        %{
        fname=sprintf('bioid_%04d',i);
        img_path=strcat(TRAIN_DATA_PATH,fname,'.pgm');
        shp_path=strcat(TRAIN_DATA_PATH,fname,'.pts');
        
        Train_Images(i+1,:,:)=loadBioIDImage(img_path); % w*h*N
        Train_GTShapes(i+1,:)=loadBioIDShape(shp_path); % 40*N
        %}
        
        fname=sprintf('%04d',i);
        img_path=strcat(TRAIN_DATA_PATH,fname,'.png');
        shp_path=strcat(TRAIN_DATA_PATH,fname,'.land');
        
        Train_Images(i+1,:,:)=loadPNG(img_path); % w*h*N
        Train_GTShapes(i+1,:)=loadShape(shp_path,width,height); % 40*N
        
    end
    
    mean_shape=mean(Train_GTShapes); % same rectangle??
    
    % Augment training data
    %{
    Train_S=zeros(TRAIN_DATA_SIZE*NUM_AUG,2*NUM_LANDMARKS); % augment to TRAIN_DATA_SIZE*NUM_AUG
    for i=1:TRAIN_DATA_SIZE
        % randperm(TRAIN_DATA_SIZE,NUM_AUG)
        %idx=randperm(TRAIN_DATA_SIZE);idx=idx(1:NUM_AUG);
        idx=randsample(TRAIN_DATA_SIZE,NUM_AUG);
        Train_S(((i-1)*NUM_AUG+1):(i*NUM_AUG),:)=Train_GTShapes(idx,:);
    end
    %}
    idx=randsample(TRAIN_DATA_SIZE,TRAIN_DATA_SIZE*NUM_AUG,true);
    Train_S=Train_GTShapes(idx,:);

end