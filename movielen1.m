% collaborative filter example with movie len data
% see Example III.1
load movielen100kdata.mat  % load the 100k movielen data (user, movie, rate, time)
ratio=.9;  % use half data for training, and half data for testing
N=max(P(:,1)); % total number of users
M=max(P(:,2)); % total number of movies
L=length(P(:,1)); % total number of data record
Lx=round(L*ratio); % data record used in training

P1=zeros(N,M); P1f=zeros(N,M); % training data record, full data record
for i=1:N,
    k=find(P(:,1)==i); % find all the user i's ratings
    P1f(i,P(k,2))=P(k,3); % full data record
    k=find(P(1:Lx,1)==i); % find user i's ratings in the training data record
    if ~isempty(k), P1(i,P(k,2))=P(k,3); end % patial data record used in training
end

%%%% P1f is the NxM full data table: N users, M moview
%%%% P1 is the NxM training data table: N users, M movies. 0 values means no rating
%%%% We use portion of the P1f non-zero values as P1 to train, and use the
%%%% other to calculate prediction errors.

%%% next: we do pre-processsing 
%%%%   (change 0 to column mean value (average rate for each movie))
%%%%   (remove row mean vlaue (average rating of each user))
P2=P1;  % save the original source data table P1, do pre-processing to P2
for i=1:M,   % fill zero-entries with column mean 
    temp=find(P1(:,i)==0);
    if length(temp)==N, continue; end
    P2(temp,i)=sum(P1(:,i))/(N-length(temp));
end
P1m=zeros(1,N);  % store each row's average
P3=P2;   % save P2, conduct mean removing on P3 only
for i=1:N    % normalize each row (each user) by removing the mean
    P1m(i)=mean(P2(i,:));
    P3(i,:)=P2(i,:)-P1m(i);
end

%%% next: we do SVD based collaborative filtering
K=14;  % number of non-zero singular values we will keep
[U,D,V]=svd(P3);      % SVD: A=U*D*V'
A=U(:,1:K)*sqrt(D(1:K,1:K));  % inverse SVD:  A1=U*D1*V'
B=sqrt(D(1:K,1:K))*V(:,1:K)';
R=A*B; % estimated recommendation score
R1=R;
for i=1:N  % denormalize: add the row average back so that we can get the ratings
    R1(i,:)=R(i,:)+P1m(i);
end

%%% finally: check prediction performance: P1f (original rating data, we use those data that not in P1)
%%% and R1 (rating data fater collaboration filtering). The major
%%% difference is that the original non-zero values in P1f (but 0 in P1) are replaced by a
%%% predicted rating.

%Code Made by Xinran Hao
R2=round(R1);
P4=zeros(N,M);
R3=zeros(N,2);
for i=1:N
    for j=1:M
        if P1f(i,j)~=0 && P1(i,j)==0 , % consider only those entries that are rated in P1f but not used in training
            P4(i,j)=P1f(i,j);   % Saving the number which is in P1f but not in P1
        end
    end
end
k=1;
error=0;
for i=1:N
    for j=1:M
        if R2(i,j)~=P4(i,j) && P4(i,j)~=0, %This is for finding the difference between the prediction and the truth.
            error=error+1;  % This is for catching the error for comparing the difference.
            R3(k,1)=i;  % This is for recording the unmatched data information into the a empty vector.
            R3(k,2)=j;
            k=k+1;
        end
    end
end

t=0;    %This code can calculate the total number in P1f but not in P1.
for i=1:N
    for j=1:M
        if P4(i,j)~=0
            t=t+1;
        end
    end
end





