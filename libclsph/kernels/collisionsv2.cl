
typedef struct {
	float3 position, next_velocity;
	int collision_happened;
	float time_elapsed;
	int indice;
} collision_response;

int respond(collision_response* response, float3 p, float3 normal,float restitution,float d, float time_elapsed) {
	//hack to avoid points directly on the faces, the collision detection code should be
	response->position = p + d*normal;

	response->next_velocity -=
		(1.f +
			restitution *
			d /
			(time_elapsed * length(response->next_velocity))) *
		dot(response->next_velocity, normal) * normal;
		//response->bob = normal;

	return 1;
}

float det(float x1, float y1,float x2, float y2){
    return x1*y2 - y1*x2;
}

float distPointDroite(float x, float y, float z, float x1, float y1, float x2, float y2){
    float A = y - x1;
    float B = z - y1;
    float C = x2 - x1;
    float D = y2 - y1;

    float dot = A * C + B * D;
    float len_sq = C * C + D * D;
    float param = -1;
    if (len_sq != 0) //in case of 0 length line
      param = dot / len_sq;

    float xx, yy;

    if (param < 0) {
        xx = x1;
        yy = y1;
    }
    else if (param > 1) {
        xx = x2;
        yy = y2;
    }
    else {
        xx = x1 + param * C;
        yy = y1 + param * D;
    }

    float dy = y - xx;
    float dz = z - yy;
    return sqrt(x*x +dz * dz + dy * dy);

}

void kernel computeDistanceField(
    global float* df,
    global const BB* bboxs,
    global const float* transforms,
    global const float* rvertices,
    uint face_count,
    uint gridcount
    ) {
        int indice =face_count-1;
        int toffset =bboxs[indice].offset;
        float temd= 20;
        const size_t current_df_index = get_global_id(0);
        while(toffset>current_df_index && indice>0){
            indice--;
            toffset = bboxs[indice].offset;
        }
        if(current_df_index<gridcount){
            int x = ((current_df_index-toffset)%(bboxs[indice].size_x*bboxs[indice].size_z))%bboxs[indice].size_x;
            int z = ((current_df_index-toffset)%(bboxs[indice].size_x*bboxs[indice].size_z))/bboxs[indice].size_x;
            int y = (current_df_index-toffset)/(bboxs[indice].size_x*bboxs[indice].size_z);

            float px = x*(bboxs[indice].maxx-bboxs[indice].minx)/(bboxs[indice].size_x-1)+bboxs[indice].minx;
            float py = y*(bboxs[indice].maxy-bboxs[indice].miny)/(bboxs[indice].size_y-1)+bboxs[indice].miny;
            float pz = z*(bboxs[indice].maxz-bboxs[indice].minz)/(bboxs[indice].size_z-1)+bboxs[indice].minz;

            for(int i=0;i<face_count;i++){
                if(px<=bboxs[i].maxx && px>=bboxs[i].minx && py<=bboxs[i].maxy && py>=bboxs[i].miny && pz<=bboxs[i].maxz && pz>=bboxs[i].minz ){
                    float tpx = px + transforms[i*12+3];
                    float tpy= py  + transforms[i*12+7];
                    float tpz=pz + transforms[i*12+11];

                    float rpx= transforms[i*12]*tpx + transforms[i*12+1]*tpy + transforms[i*12+2]*tpz ;
                    float rpy= transforms[i*12+4]*tpx + transforms[i*12+5]*tpy + transforms[i*12+6]*tpz;
                    float rpz= transforms[i*12+8]*tpx + transforms[i*12+9]*tpy + transforms[i*12+10]*tpz;

                    float v1y = rvertices[4*i+1];
                    float v2x = rvertices[4*i+2];
                    float v2y = rvertices[4*i+3];

                    float a=det(rpy,rpz,0,v1y)/det(v2x,v2y,0,v1y);

                    float b=-det(rpy,rpz,v2x,v2y)/det(v2x,v2y,0,v1y);

                    float d, td;
                    if(a>0 && b>0 && a+b<1)
                        d= fabs(rpx);
                    else{
                        d=distPointDroite(rpx,rpy,rpz,0,0,rvertices[4*i],rvertices[4*i+1]);
                        td = distPointDroite(rpx,rpy,rpz,rvertices[4*i],rvertices[4*i+1],rvertices[4*i+2],rvertices[4*i+3]);
                        if(td<d){
                            d=td;
                        }
                        td=distPointDroite(rpx,rpy,rpz,0,0,rvertices[4*i+2],rvertices[4*i+3]);
                        if(td<d){
                            d=td;
                        }
                    }
                    if(d<fabs(temd)){
                        temd=copysign(d,rpx);
                    }

                }
            }

            df[current_df_index]= temd;
        }
    }

float weigthedAverage(float x, float x1 , float x2,float d1, float d2){
    return ((x2-x)/(x2-x1))*d1+((x-x1)/(x2-x1))*d2;
}

float bilinearInterpolation(float x, float y, float xmin , float ymin, float xmax, float ymax, float d00, float d01, float d10, float d11){
    float R1 = weigthedAverage(x,xmin,xmax,d00,d10);
    float R2 = weigthedAverage(x,xmin,xmax,d01,d11);
    return weigthedAverage(y,ymin,ymax,R1,R2);
}

int getDFindex(BB bbox,float x, float y, float z, short a, short b, short c){
    return bbox.offset + (y+b)*bbox.size_x*bbox.size_z+bbox.size_x*(z+c)+x+a;
}

collision_response handle_collisions(float3 old_position,
	float3 position,
	float3 next,
	float restitution, float time_elapsed,
	global const float* df,
	global const BB* bboxs,
	uint face_count) {
        int indice =-1;
        collision_response response = {
            position, next, 0, time_elapsed,-1
        };
        const size_t current_df_index = get_global_id(0);
        for(int i=0;i<face_count;i++){
            if(position.x<=bboxs[i].maxx && position.x>=bboxs[i].minx && position.y<=bboxs[i].maxy && position.y>=bboxs[i].miny && position.z<=bboxs[i].maxz && position.z>=bboxs[i].minz ){
                indice = i;
            }
        }

        if(indice>-1){
            float sidex = (bboxs[indice].maxx-bboxs[indice].minx)/(bboxs[indice].size_x-1);
            float sidey = (bboxs[indice].maxy-bboxs[indice].miny)/(bboxs[indice].size_y-1);
            float sidez = (bboxs[indice].maxz-bboxs[indice].minz)/(bboxs[indice].size_z-1);

            int x = (position.x - bboxs[indice].minx)/(sidex);
            int y = (position.y - bboxs[indice].miny)/(sidey);
            int z = (position.z - bboxs[indice].minz)/(sidez);

            float bx = x*sidex+bboxs[indice].minx;
            float by = y*sidey+bboxs[indice].miny;
            float bz = z*sidez+bboxs[indice].minz;

            float facedown = bilinearInterpolation(position.x,position.z, bx,bz,bx+sidex,bz+sidez,df[getDFindex(bboxs[indice],x,y,z,0,0,0)],df[getDFindex(bboxs[indice],x,y,z,0,0,1)],df[getDFindex(bboxs[indice],x,y,z,1,0,0)],df[getDFindex(bboxs[indice],x,y,z,1,0,1)]);
            float faceup =   bilinearInterpolation(position.x,position.z, bx,bz,bx+sidex,bz+sidez,df[getDFindex(bboxs[indice],x,y,z,0,1,0)],df[getDFindex(bboxs[indice],x,y,z,0,1,1)],df[getDFindex(bboxs[indice],x,y,z,1,1,0)],df[getDFindex(bboxs[indice],x,y,z,1,1,1)]);

            float d = weigthedAverage(position.y,by,by+sidey,facedown,faceup);
            response.indice = indice;
            if(d<0.02){
                response.collision_happened = 1;
                response.indice = indice*10;
                float faceright  = bilinearInterpolation(position.y,position.z, by,bz,by+sidey,bz+sidez,df[getDFindex(bboxs[indice],x,y,z,1,0,0)],df[getDFindex(bboxs[indice],x,y,z,1,0,1)],df[getDFindex(bboxs[indice],x,y,z,1,1,0)],df[getDFindex(bboxs[indice],x,y,z,1,1,1)]);
                float faceleft =  bilinearInterpolation(position.y,position.z, by,bz,by+sidey,bz+sidez,df[getDFindex(bboxs[indice],x,y,z,0,0,0)],df[getDFindex(bboxs[indice],x,y,z,0,0,1)],df[getDFindex(bboxs[indice],x,y,z,0,1,0)],df[getDFindex(bboxs[indice],x,y,z,0,1,1)]);

                float faceback = bilinearInterpolation(position.x,position.y, bx,by,bx+sidex,by+sidey,df[getDFindex(bboxs[indice],x,y,z,0,0,0)],df[getDFindex(bboxs[indice],x,y,z,0,1,0)],df[getDFindex(bboxs[indice],x,y,z,1,0,0)],df[getDFindex(bboxs[indice],x,y,z,1,1,0)]);
                float facefront = bilinearInterpolation(position.x,position.y, bx,by,bx+sidex,by+sidey,df[getDFindex(bboxs[indice],x,y,z,0,0,1)],df[getDFindex(bboxs[indice],x,y,z,0,1,1)],df[getDFindex(bboxs[indice],x,y,z,1,0,1)],df[getDFindex(bboxs[indice],x,y,z,1,1,1)]);

                float3 normal = {
                    (faceright- faceleft),
                    (faceup-facedown),
                    (facefront-faceback)
                };
                float lenn = length(normal);
                normal/=lenn;

                respond(&response, position, normal, restitution,fabs(d), time_elapsed);
                response.time_elapsed = time_elapsed *
                  (length(response.position - old_position) / length(position - old_position));

            }
        }

        return response;
    }

