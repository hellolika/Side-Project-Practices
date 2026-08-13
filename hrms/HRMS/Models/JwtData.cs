using HRMS.Models.Responses;

namespace HRMS.Models
{
    public class JwtData
    {
        public int Id { get; set; }
        public string Username { get; set; } = string.Empty;
        public int Permission { get; set; } = 0;
        public List<string> Permissions { get; set; }
        public bool IsAdmin()
        {
            return Permission >= 2;
        }

        public JwtData()
        {
        }

        public JwtData(LoginResponse response,IEnumerable<MemberPermissionResponse> permissionResponse)
        {
            Id = response.MemberId;
            Username = response.Username;
            Permission = response.Permission;
            Permissions = permissionResponse.Select(m => m.PermissionId.ToString())
                .ToList();

        }
    }
}