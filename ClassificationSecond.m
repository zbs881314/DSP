I=200; X=zeros(2,I); Y=zeros(1,I);
s = struct;
for i=1:I,
 t=rand(1);
 if rand(1)<0.5,
     X(:,i)=[t; .75*cos(2*pi*t)-0.5]; 
     Y(i)=1;
 else
     X(:,i)=[t; .75*cos(2*pi*t)+0.5];
     Y(i)=0;
 end
end
classifier=fitcnb(X(:,1:100)', Y(1:100)');
lable=predict(classifier,X(:,101:200)');
error=0;
for i=1:100
    if Y(100+i)~=lable(i)
        error=error+1;
    end
end
