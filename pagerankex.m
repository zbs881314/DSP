% pagerank exercise
% see Example III.2
% for a special 4-node network

% adjacency matrix (connection matrix)
A=[0 1 1 0 0 0; 0 0 1 0 0 0; 1 0 0 0 0 0 ; 1 0 1 0 0 0; 0 0 1 0 0 0; 0 1 0 0 1 0];
N=length(A(:,1)); % total number of nodes
alpha=0.23;  
x=rand(N,1);  % random initialization of page rank values

T=100000;  % total number of upating iterations
Y=zeros(N,T); % store page rank values in each iteration
for i=1:T-1, % update page rank values
    Y(:,i)=x;
    x=alpha+(1-alpha)*A*x;
end
Y(:,T)=x;

x'
plot(Y(1,:)), xlabel('iteration'), ylabel('PageRank value'),grid
