using HRMS.Enum;
using HRMS.Filters;
using HRMS.Models;
using HRMS.Models.Requests;
using HRMS.Models.Responses;
using HRMS.Services.Interfaces;
using Microsoft.AspNetCore.Mvc;

namespace HRMS.Controllers
{
    [Route("/api/[controller]")]
    [ServiceFilter(typeof(AuthenticateJwt))]
    [ApiController]
    public class DataController : Controller
    {
        private readonly IMemberService _memberService;

        public DataController(IMemberService memberService)
        {
            _memberService = memberService;
        }
        
        [HasPermission(PermissionEnum.CanGetLeaveType)]
        [HttpGet("GetLeaveType")]
        public ApiBaseResponse<List<LeaveType>> GetAllLeaveType()
        {
            return _memberService.GetAllLeaveType();
        }
        
        [HasPermission(PermissionEnum.CanGetTeam)]
        [HttpGet("GetTeam")]
        public ApiBaseResponse<List<TeamInfo>> GetAllTeamInfo()
        {
            return _memberService.GetAllTeamInfo();
        }
        
        [HasPermission(PermissionEnum.CanGetLocation)]
        [HttpGet("GetLocation")]
        public ApiBaseResponse<List<LocationDetails>> GetLocation()
        {
            return _memberService.GetLocation();
        }

        [HasPermission(PermissionEnum.CanGetAnnouncements)]
        [HttpPost("GetAnnouncements")]
        public ApiBaseResponse<GetAnnouncementResponse> GetAnnouncements(GetAnnouncementRequest request)
        {
            return _memberService.GetAnnouncements(request);
        }
        
        [HttpGet("GetAvailableSettingForMemberOperation")]
        public ApiBaseResponse<GetAvailableSettingForMemberOperationResponse> GetAvailableSettingForMemberOperation()
        {
            return _memberService.GetAvailableSettingForMemberOperation();
        }
    }
}