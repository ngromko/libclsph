#define KERNEL_INCLUDE

void kernel fillUintArray(global uint* bob, uint value, uint length) {
  uint id = get_global_id(0);
  if (id < length) bob[id] = value;
}
