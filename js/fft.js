  function makeFFT(n){
    const cos = new Float64Array(n/2), sin = new Float64Array(n/2), rev = new Uint32Array(n);
    for (let i=0;i<n/2;i++){ cos[i]=Math.cos(2*Math.PI*i/n); sin[i]=Math.sin(2*Math.PI*i/n); }
    const bits = Math.log2(n);
    for (let i=0;i<n;i++){ let r=0, x=i; for (let b=0;b<bits;b++){ r=(r<<1)|(x&1); x>>=1; } rev[i]=r; }
    return (re, im) => {
      for (let i=0;i<n;i++){ const j=rev[i]; if (j>i){ let t=re[i]; re[i]=re[j]; re[j]=t; t=im[i]; im[i]=im[j]; im[j]=t; } }
      for (let size=2; size<=n; size<<=1){
        const half=size>>1, step=n/size;
        for (let i=0;i<n;i+=size){
          for (let j=i,k=0;j<i+half;j++,k+=step){
            const l=j+half, tre=re[l]*cos[k]+im[l]*sin[k], tim=-re[l]*sin[k]+im[l]*cos[k];
            re[l]=re[j]-tre; im[l]=im[j]-tim; re[j]+=tre; im[j]+=tim;
          }
        }
      }
    };
  }


module.exports = makeFFT;
